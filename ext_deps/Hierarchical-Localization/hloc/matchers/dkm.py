import warnings

import torch
import torchvision.transforms as T
from dkm import DKMv3_indoor, DKMv3_outdoor
from PIL import Image

from ..utils.base_model import BaseModel


class DKM(BaseModel):
    default_conf = {
        "weights": "outdoor",
        "sample_num": 10000,
        "match_threshold": 0.2,
        "max_num_matches": None,
    }
    required_inputs = ["image0", "image1"]

    def _init(self, conf):
        if conf["weights"] == "outdoor":
            self.net = DKMv3_outdoor()
            self.W = 720
            self.H = 540
        elif conf["weights"] == "indoor":
            self.net = DKMv3_indoor()
            self.W = 640
            self.H = 480
        self.net.sample_thresh = conf["match_threshold"]
        self.img_transformer = T.ToPILImage("RGB")

    def _forward(self, data):
        device = data["image0"].device
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            im1 = self.img_transformer(data["image0"].squeeze())
            im2 = self.img_transformer(data["image1"].squeeze())

            warp, certainty = self.net.match(
                im1.resize((self.W, self.H)),
                im2.resize((self.W, self.H)),
                device=device,
            )

            kmatches, certainty_ = self.net.sample(
                warp,
                certainty,
                num=self.conf["sample_num"],
                device=device,
            )

        pred = {}

        kpts1 = kmatches[:, :2]
        kpts2 = kmatches[:, 2:]

        kpts1, kpts2 = self.net.to_pixel_coordinates(
            kmatches, im1.size[1], im1.size[0], im2.size[1], im2.size[0]
        )

        top_k = self.conf["max_num_matches"]
        if top_k is not None and len(certainty_) > top_k:
            keep = torch.argsort(certainty_, descending=True)[:top_k]
            pred["keypoints0"], pred["keypoints1"] = kpts1[keep, :], kpts2[keep, :]
            pred["scores"] = certainty_[keep]
        else:
            pred["keypoints0"], pred["keypoints1"] = kpts1, kpts2
            pred["scores"] = certainty_

        return pred
