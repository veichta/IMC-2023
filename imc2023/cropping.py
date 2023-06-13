import cv2
import numpy as np
from tqdm import tqdm
from typing import List, Tuple

from hloc import extract_features, match_features, pairs_from_retrieval
from hloc.utils.io import list_h5_names, get_matches, get_keypoints

from imc2023.configs import configs
from imc2023.utils.utils import DataPaths


def get_iou(box1: Tuple[int], box2: Tuple[int]) -> float:
    intersection_area = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0])) * max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area


def crop_is_valid(
    new_bb: Tuple[int],
    previous_bbs: List[Tuple[int]],
    max_iou: float = 0.5,
) -> bool:
    for bb in previous_bbs:
        if get_iou(new_bb, bb) > max_iou:
            return False
    return True


def crop_images(
    paths: DataPaths,
    img_list: List[str],
    min_rel_crop_size: float,
    max_rel_crop_size: float,
) -> List[str]:
    """Crop image and add them to the image directory. Return updated image list.

    Args:
        paths (DataPaths): Data paths.
        min_rel_crop_size (float): Crops must have a larger relative size
        max_rel_crop_size (float): Crops must have a smaller relative size
    """
    # get top 10 netvlad pairs
    extract_features.main(
        conf=extract_features.confs["netvlad"],
        image_dir=paths.image_dir,
        feature_path=paths.cropping_features_retrieval,
    )
    pairs_from_retrieval.main(
        descriptors=paths.cropping_features_retrieval,
        num_matched=10,
        output=paths.cropping_pairs_path,
    )

    # extract and match features using SIFT and NN-ratio
    extract_features.main(
        conf=configs["SIFT"]["features"],
        image_dir=paths.image_dir,
        feature_path=paths.cropping_features_path,
    )
    match_features.main(
        conf=configs["SIFT"]["matches"],
        pairs=paths.cropping_pairs_path,
        features=paths.cropping_features_path,
        matches=paths.cropping_matches_path,
    )

    bounding_boxes = {} # key: image name, value: list of tuples of bb coords

    # iterate through candidate pairs and create crops
    pairs = list(list_h5_names(paths.cropping_matches_path))
    for pair in tqdm(pairs, desc="Processing pairs...", ncols=80):
        img_1, img_2 = pair.split("/")

        # get keypoints and matches
        kp_1 = get_keypoints(paths.cropping_features_path, img_1).astype(np.int32)
        kp_2 = get_keypoints(paths.cropping_features_path, img_2).astype(np.int32)
        matches, scores = get_matches(paths.cropping_matches_path, img_1, img_2)

        if len(matches) < 100:
            continue  # too few matches

        # get top 80% matches
        threshold = np.quantile(scores, 0.2)
        mask = scores >= threshold
        top_matches = matches[mask]

        # compute bounding boxes based on the keypoints of the top 80% matches
        top_kp_1 = kp_1[top_matches[:, 0]]
        top_kp_2 = kp_2[top_matches[:, 1]]
        # (x_min, y_min, x_max, y_max)
        bb_1 = (top_kp_1[:, 0].min(), top_kp_1[:, 1].min(), top_kp_1[:, 0].max(), top_kp_1[:, 1].max())
        bb_2 = (top_kp_2[:, 0].min(), top_kp_2[:, 1].min(), top_kp_2[:, 0].max(), top_kp_2[:, 1].max())

        # crop original images
        original_image_1 = cv2.imread(str(paths.image_dir / img_1))
        cropped_image_1 = original_image_1[
            bb_1[1] : bb_1[3] + 1,
            bb_1[0] : bb_1[2] + 1,
        ]
        original_image_2 = cv2.imread(str(paths.image_dir / img_2))
        cropped_image_2 = original_image_2[
            bb_2[1] : bb_2[3] + 1,
            bb_2[0] : bb_2[2] + 1,
        ]

        # save crops if relative size and IoU conditions are fulfilled
        rel_size_1 = cropped_image_1.size / original_image_1.size
        rel_size_2 = cropped_image_2.size / original_image_2.size

        if rel_size_1 >= min_rel_crop_size and rel_size_1 <= max_rel_crop_size:
            if img_1 not in bounding_boxes or crop_is_valid(bb_1, bounding_boxes[img_1]):
                # save new bb coords
                if img_1 not in bounding_boxes:
                    bounding_boxes[img_1] = [bb_1]
                else:
                    bounding_boxes[img_1].append(bb_1)

                # save new crop
                crop_name = f"{img_1}_crop_{len(bounding_boxes[img_1])}.jpg"
                cv2.imwrite(str(paths.image_dir / crop_name), cropped_image_1)

                # add new image to image list
                img_list.append(crop_name)

        if rel_size_2 >= min_rel_crop_size and rel_size_2 <= max_rel_crop_size:
            if img_2 not in bounding_boxes or crop_is_valid(bb_2, bounding_boxes[img_2]):
                # save new bb coords
                if img_2 not in bounding_boxes:
                    bounding_boxes[img_2] = [bb_2]
                else:
                    bounding_boxes[img_2].append(bb_2)

                # save new crop
                crop_name = f"{img_2}_crop_{len(bounding_boxes[img_2])}.jpg"
                cv2.imwrite(str(paths.image_dir / crop_name), cropped_image_2)

                # add new image to image list
                img_list.append(crop_name)

    return img_list
