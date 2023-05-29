import logging
import cv2
import h5py
import numpy as np
from tqdm import tqdm
from typing import Any, Dict

from hloc import extract_features, match_features
from hloc.utils.io import list_h5_names, get_matches, get_keypoints

from imc2023.utils.utils import DataPaths


def crop_matching(
    paths: DataPaths, 
    config: Dict[str, Any], 
    min_rel_crop_size: float,
    max_rel_crop_size: float,
    is_ensemble: bool, 
    sep: str = "-+-",
) -> None:
    """Perform feature matching on cropped images and add new matches to the current ones.

    Args:
        paths (DataPaths): Data paths.
        config (Dict[str, Any]): Configs of the current run.
        min_rel_crop_size (float): BOTH crops must have a larger relative size
        max_rel_crop_size (float): EITHER crop must have a smaller relative size
        is_ensemble (bool): Whether the current run is using an ensemble.
        sep (str, optional): Separator when creating filenames for cropped images. Defaults to "-+-".
    """    
    logging.info("Creating crops for all matches")

    # new list of pairs for the matching on crops
    crop_pairs = []

    # dictionary of offsets to transform the keypoints from "crop spaces" to the original image spaces
    offsets = {}

    # iterate through all original pairs and create crops
    original_pairs = list(list_h5_names(paths.matches_path))
    for pair in tqdm(original_pairs):
        img_1, img_2 = pair.split("/")

        # get original keypoints and matches
        kp_1 = get_keypoints(paths.features_path, img_1).astype(np.int32)
        kp_2 = get_keypoints(paths.features_path, img_2).astype(np.int32)
        matches, scores = get_matches(paths.matches_path, img_1, img_2)

        if len(matches) < 100:
            continue # too few matches

        # get top 80% matches
        threshold = np.quantile(scores, 0.2)
        mask = scores >= threshold
        top_matches = matches[mask]

        # compute bounding boxes based on the keypoints of the top 80% matches
        top_kp_1 = kp_1[top_matches[:,0]]
        top_kp_2 = kp_2[top_matches[:,1]]
        original_image_1 = cv2.imread(str(paths.image_dir / img_1))
        original_image_2 = cv2.imread(str(paths.image_dir / img_2))
        cropped_image_1 = original_image_1[
            top_kp_1[:, 1].min() : top_kp_1[:, 1].max() + 1, 
            top_kp_1[:, 0].min() : top_kp_1[:, 0].max() + 1, 
        ]
        cropped_image_2 = original_image_2[
            top_kp_2[:, 1].min() : top_kp_2[:, 1].max() + 1, 
            top_kp_2[:, 0].min() : top_kp_2[:, 0].max() + 1, 
        ]

        # check if the relative size conditions are fulfilled
        rel_size_1 = cropped_image_1.size / original_image_1.size
        rel_size_2 = cropped_image_2.size / original_image_2.size

        if rel_size_1 <= min_rel_crop_size or rel_size_2 < min_rel_crop_size:
            # one of the crops or both crops are too small ==> avoid degenerate crops
            continue 

        if rel_size_1 >= max_rel_crop_size and rel_size_2 >= max_rel_crop_size:
            # both crops are almost the same size as the original images
            # ==> crops are not useful (almost same matches as on the original images)
            continue

        # define new names for the crops based on the current pair because each 
        # original image will be cropped in a different way for each original matching
        # ==> use distinctive separator to simplify parsing during the aggregation!
        name_1 = f"{img_1}{sep}{img_2}{sep}1.jpg" 
        name_2 = f"{img_1}{sep}{img_2}{sep}2.jpg"

        # save crops
        cv2.imwrite(str(paths.cropped_image_dir / name_1), cropped_image_1)
        cv2.imwrite(str(paths.cropped_image_dir / name_2), cropped_image_2)

        # create new matching pair and save offsets for image space transformations
        crop_pairs.append((name_1, name_2))
        offsets[name_1] = (top_kp_1[:, 0].min(), top_kp_1[:, 1].min())
        offsets[name_2] = (top_kp_2[:, 0].min(), top_kp_2[:, 1].min())

    # save new list of crop pairs
    with open(paths.cropped_pairs_path, "w") as f:
        for p1, p2 in crop_pairs:
            f.write(f"{p1} {p2}\n")
    
    logging.info("Performing feature extraction and matching on crops")
    extract_features.main(
        conf=config["features"][0] if is_ensemble else config["features"],
        image_dir=paths.cropped_image_dir,
        feature_path=paths.cropped_features_path,
    )
    match_features.main(
        conf=config["matches"][0] if is_ensemble else config["matches"],
        pairs=paths.cropped_pairs_path,
        features=paths.cropped_features_path,
        matches=paths.cropped_matches_path,
    )

    logging.info("Transforming keypoints from cropped image spaces to original image spaces")
    with h5py.File(str(paths.cropped_features_path), "r+", libver="latest") as f:
        for name in offsets.keys():
            keypoints = f[name]["keypoints"].__array__()
            keypoints[:,0] += offsets[name][0]
            keypoints[:,1] += offsets[name][1]
            f[name]["keypoints"][...] = keypoints

    logging.info("Concatenating features and matches from crops with original features and matches")
    # TODO
