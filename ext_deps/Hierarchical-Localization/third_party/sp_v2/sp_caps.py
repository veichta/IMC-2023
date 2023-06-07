"""
Retrained version of SuperPoint, a feature detector and descriptor.

Described in:
    SuperPoint: Self-Supervised Interest Point Detection and Description,
    Daniel DeTone, Tomasz Malisiewicz, Andrew Rabinovich, CVPRW 2018.

Original code: github.com/MagicLeapResearch/SuperPointPretrainedNetwork
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from kornia.geometry.transform import warp_perspective
from kornia.morphology import erosion
from torch import Tensor
from typing import List, Optional
from pathlib import Path

from omegaconf import OmegaConf

# from gluefactory.models.base_model import BaseModel
# from gluefactory.datasets.homographies import sample_homography_corners


def simple_nms(scores, radius):
    """Perform non maximum suppression on the heatmap using max-pooling.
    This method does not suppress contiguous points that have the same score.
    Args:
        scores: the score heatmap of size `(B, H, W)`.
        size: an interger scalar, the radius of the NMS window.
    """
    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=radius*2+1, stride=1, padding=radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def remove_borders(keypoints, b, h, w):
    mask_h = (keypoints[1] >= b) & (keypoints[1] < (h - b))
    mask_w = (keypoints[2] >= b) & (keypoints[2] < (w - b))
    mask = mask_h & mask_w
    return (keypoints[0][mask], keypoints[1][mask], keypoints[2][mask])


def top_k_keypoints(keypoints, scores, k):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0, sorted=True)
    return keypoints[indices], scores


def soft_argmax_refinement(keypoints, scores, radius: int):
    width = 2*radius + 1
    sum_ = torch.nn.functional.avg_pool2d(
        scores[:, None], width, 1, radius, divisor_override=1)
    sum_ = torch.clamp(sum_, min=1e-6)
    ar = torch.arange(-radius, radius+1).to(scores)
    kernel_x = ar[None].expand(width, -1)[None, None]
    dx = torch.nn.functional.conv2d(
        scores[:, None], kernel_x, padding=radius)
    dy = torch.nn.functional.conv2d(
        scores[:, None], kernel_x.transpose(2, 3), padding=radius)
    dydx = torch.stack([dy[:, 0], dx[:, 0]], -1) / sum_[:, 0, :, :, None]
    refined_keypoints = []
    for i, kpts in enumerate(keypoints):
        delta = dydx[i][tuple(kpts.t())]
        refined_keypoints.append(kpts.float() + delta)
    return refined_keypoints


# The original keypoint sampling is incorrect. We patch it here but
# keep the original one above for legacy.
def sample_descriptors_fix_sampling(keypoints, descriptors, s: int = 8,
                                    normalize=True):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    keypoints = keypoints / (keypoints.new_tensor([w, h]) * s)
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2),
        mode='bilinear', align_corners=False).reshape(b, c, -1)
    if normalize:
        descriptors = torch.nn.functional.normalize(
            descriptors, p=2, dim=1)
    return descriptors.permute(0, 2, 1)


class Bottleneck(nn.Module):
    # Bottleneck inspired from "Deep residual learning for image recognition"
    # https://arxiv.org/abs/1512.03385.

    expansion: int = 2
    norm_layer = nn.BatchNorm2d

    def __init__(self, inplanes: int, planes: int,
                 downsample: Optional[nn.Module] = None,) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, padding=0)
        self.bn1 = self.norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.bn2 = self.norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion,
                               kernel_size=1, padding=0)
        self.bn3 = self.norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetCustom(torch.nn.Module):
    def __init__(self, blocks: List[int] = [2, 2, 2], grayscale: bool = False,
                 zero_init_residual: bool = True):
        super().__init__()
        # Stack 3 consecutive blocks of ResNet bottlenecks,
        # with 3 downsamplings performed with max pooling
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1 if grayscale else 3, self.inplanes,
                               kernel_size=7, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.conv2 = nn.Conv2d(self.inplanes, self.inplanes,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(self.inplanes)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)

        # Backbone
        self.layer1 = self._make_layer(64, blocks[0])
        self.layer2 = self._make_layer(128, blocks[1])
        self.layer3 = self._make_layer(128, blocks[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros,
        # and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, planes: int, num_blocks: int) -> nn.Sequential:
        block = Bottleneck
        downsample = None
        if self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, padding=0),
                block.norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.maxpool(x)

        x = self.layer2(x)
        x = self.maxpool(x)

        x = self.layer3(x)

        return x


class SP_CAPS(nn.Module):
    default_conf = {
        'backbone': {
            'blocks': [2, 2, 2]
        },
        'has_detector': True,
        'has_descriptor': True,

        # Inference
        'sparse_outputs': True,
        'nms_radius': 8,
        'refinement_radius': 3,
        'detection_threshold': 0.000,
        'max_num_keypoints': -1,
        'force_num_keypoints': False,
        'remove_borders': 4,

        # Descriptor
        'descriptor_dim': 128,

        'weights': None,

        # Homography adaptation
        'ha': {
            'enable': False,
            'num_H': 10,
            'mini_bs': 5,
            'aggregation': 'mean',
            'H_params': {
                'difficulty': 0.8,
                'translation': 1.0,
                'max_angle': 60,
                'n_angles': 10,
                'min_convexity': 0.05
            },
        },
    }
    required_data_keys = ['image']

    def __init__(self, conf):
        super().__init__()
        self.conf = conf = OmegaConf.merge(self.default_conf, conf)
        # Backbone network
        self.backbone = ResNetCustom(self.conf.backbone.blocks)
        dense_feat_dim = 256

        if conf.has_detector:
            self.convPa = nn.Conv2d(dense_feat_dim, 256, kernel_size=3,
                                    stride=1, padding=1)
            self.convPb = nn.Conv2d(256, 65, kernel_size=1,
                                    stride=1, padding=0)

        if conf.has_descriptor:
            self.convDa = nn.Conv2d(dense_feat_dim, 256, kernel_size=3,
                                    stride=1, padding=1)
            self.convDb = nn.Conv2d(256, conf.descriptor_dim, kernel_size=1,
                                    stride=1, padding=0)

        self.erosion_kernel = torch.tensor(
            [[0, 0, 1, 0, 0],
             [0, 1, 1, 1, 0],
             [1, 1, 1, 1, 1],
             [0, 1, 1, 1, 0],
             [0, 0, 1, 0, 0]],
            dtype=torch.float
        )

        if conf.weights is not None:
            path = Path(__file__).parent
            path = path / f'{conf.weights}.tar'
            state_dict = torch.load(str(path), map_location='cpu')['model']
            self.load_state_dict({k.replace('extractor.', '') : v for k, v in state_dict.items()})

    def kp_head(self, feat):
        # Compute the dense keypoint scores
        cPa = F.relu(self.convPa(feat))
        logits = self.convPb(cPa)
        scores = torch.nn.functional.softmax(logits, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)
        return scores, logits

    def forward(self, data):
        image = data['image']

        # Shared Encoder
        x = self.backbone(image)

        pred = {}
        if self.conf.has_detector and self.conf.max_num_keypoints != 0:
            # Heatmap prediction
            if self.conf.ha.enable:
                dense_scores = self.homography_adaptation(image)
            else:
                dense_scores, logits = self.kp_head(x)
                pred['logits'] = logits
            pred['dense_score'] = dense_scores
        if self.conf.has_descriptor:
            # Compute the dense descriptors
            dense_desc = F.relu(self.convDa(x))
            dense_desc = self.convDb(dense_desc)
            dense_desc = torch.nn.functional.normalize(dense_desc, p=2, dim=1)
            pred['dense_desc'] = dense_desc

        if self.conf.sparse_outputs and self.conf.has_detector:
            sparse_pred = self.get_sparse_outputs(
                image, dense_scores,
                dense_desc=(dense_desc if self.conf.has_descriptor else None))
            pred = {**pred, **sparse_pred}

        return pred

    def get_sparse_outputs(self, image, dense_scores, dense_desc=None):
        """ Extract sparse feature points from dense scores and descriptors. """
        b_size, _, h, w = image.shape
        device = image.device
        pred = {}

        if self.conf.max_num_keypoints == 0:
            pred['keypoints'] = torch.empty(b_size, 0, 2, device=device)
            pred['keypoint_scores'] = torch.empty(b_size, 0, device=device)
            pred['descriptors'] = torch.empty(
                b_size, self.conf.descriptor_dim, 0, device=device)
            return pred

        scores = simple_nms(dense_scores, self.conf.nms_radius)

        # Extract keypoints
        best_kp = torch.where(scores > self.conf.detection_threshold)

        # Discard keypoints near the image borders
        best_kp = remove_borders(best_kp, self.conf.remove_borders,
                                 h * 8, w * 8)
        scores = scores[best_kp]

        # Separate into batches
        keypoints = [torch.stack(best_kp[1:3], dim=-1)[best_kp[0] == i]
                     for i in range(b_size)]
        scores = [scores[best_kp[0] == i] for i in range(b_size)]

        # Keep the k keypoints with highest score
        if self.conf.max_num_keypoints > 0:
            keypoints, scores = list(zip(*[
                top_k_keypoints(k, s, self.conf.max_num_keypoints)
                for k, s in zip(keypoints, scores)]))
            keypoints, scores = list(keypoints), list(scores)

        if self.conf['refinement_radius'] > 0:
            keypoints = soft_argmax_refinement(
                keypoints, dense_scores, self.conf['refinement_radius'])

        # Convert (h, w) to (x, y) and shift to the center of the pixel
        keypoints = [torch.flip(k, [1]).float() + 0.5 for k in keypoints]

        if self.conf.force_num_keypoints:
            assert self.conf.max_num_keypoints > 0
            scores = list(scores)
            for i in range(len(keypoints)):
                k, s = keypoints[i], scores[i]
                missing = self.conf.max_num_keypoints - len(k)
                if missing > 0:
                    new_k = torch.rand(missing, 2).to(k)
                    new_k = new_k * k.new_tensor([[w-1, h-1]])
                    new_s = torch.zeros(missing).to(s)
                    keypoints[i] = torch.cat([k, new_k], 0)
                    scores[i] = torch.cat([s, new_s], 0)

        if (len(keypoints) == 1) or self.conf.force_num_keypoints:
            keypoints = torch.stack(keypoints, 0)
            scores = torch.stack(scores, 0)

        pred['keypoints'] = keypoints
        pred['keypoint_scores'] = scores

        if self.conf.has_descriptor:
            # Extract descriptors
            if (len(keypoints) == 1) or self.conf.force_num_keypoints:
                # Batch sampling of the descriptors
                pred['descriptors'] = sample_descriptors_fix_sampling(
                    keypoints, dense_desc, 8)
            else:
                pred['descriptors'] = [sample_descriptors_fix_sampling(
                    k[None], d[None], 8)[0]
                                       for k, d in zip(keypoints, dense_desc)]

        return pred

    def homography_adaptation(self, img):
        """ Perform homography adaptation on the score heatmap. """
        bs = self.conf.ha.mini_bs
        num_H = self.conf.ha.num_H
        device = img.device
        self.erosion_kernel = self.erosion_kernel.to(device)
        B, _, h, w = img.shape

        # Generate homographies
        Hs = []
        for i in range(num_H):
            if i == 0:
                # Always include at least the identity
                Hs.append(torch.eye(3, dtype=torch.float, device=device))
            else:
                Hs.append(torch.tensor(
                    sample_homography_corners(
                        (w, h), patch_shape=(w, h),
                        **self.conf.ha.H_params)[0],
                    dtype=torch.float, device=device))
        Hs = torch.stack(Hs, dim=0)

        # Loop through all mini batches
        n_mini_batch = int(np.ceil(num_H / bs))
        scores = torch.empty((B, 0, h, w), dtype=torch.float, device=device)
        counts = torch.empty((B, 0, h, w), dtype=torch.float, device=device)
        for i in range(n_mini_batch):
            H = Hs[i*bs:(i+1)*bs]
            nh = len(H)
            H = H.repeat(B, 1, 1)

            # Warp the image
            warped_imgs = warp_perspective(
                torch.repeat_interleave(img, nh, dim=0),
                H, (h, w), mode='bilinear')

            # Forward pass
            with torch.no_grad():
                score = self.kp_head(self.backbone(warped_imgs))[0]

                # Compute valid pixels
                H_inv = torch.inverse(H)
                count = warp_perspective(
                    torch.ones_like(score).unsqueeze(1),
                    H, (h, w), mode='nearest')
                count = erosion(count, self.erosion_kernel)
                count = warp_perspective(count, H_inv, (h, w),
                                         mode='nearest')[:, 0]

                # Warp back the scores
                score = warp_perspective(score[:, None], H_inv, (h, w),
                                         mode='bilinear')[:, 0]

            # Aggregate the results
            scores = torch.cat([scores, score.reshape(B, nh, h, w)], dim=1)
            counts = torch.cat([counts, count.reshape(B, nh, h, w)], dim=1)

        # Aggregate the results
        if self.conf.ha.aggregation == 'mean':
            score = (scores * counts).sum(dim=1) / counts.sum(dim=1)
        elif self.conf.ha.aggregation == 'median':
            scores[counts == 0] = np.nan
            score = torch.nanmedian(scores, dim=1)[0]
        elif self.conf.ha.aggregation == 'max':
            scores[counts == 0] = 0
            score = scores.max(dim=1)[0]
        else:
            raise ValueError("Unknown aggregation method: "
                             + self.conf.ha.aggregation)
        return score

    def loss(self, pred, data):
        raise NotImplementedError

    def metrics(self, pred, data):
        raise NotImplementedError


__main_model__ = SP_CAPS