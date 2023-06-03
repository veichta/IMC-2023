import cv2
import os
import h5py
import shutil
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
) -> None:
    """Perform feature matching on cropped images and add new matches to the current ones.

    Args:
        paths (DataPaths): Data paths.
        config (Dict[str, Any]): Configs of the current run.
        min_rel_crop_size (float): BOTH crops must have a larger relative size
        max_rel_crop_size (float): EITHER crop must have a smaller relative size
        is_ensemble (bool): Whether the current run is using an ensemble.
    """
    # dictionaries that will contain all final kepoints and matches in the end
    keypoints_dict: Dict[str, np.ndarray] = {}  # only storing keypoints
    matches_dict: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}  # storing matches and scores

    # initialize keypoints dictionary with original keypoints
    with h5py.File(str(paths.features_path), "r", libver="latest") as f:
        for img in f.keys():
            keypoints_dict[img] = get_keypoints(paths.features_path, img)

    # initialize matches dictionary with original matches
    with h5py.File(str(paths.matches_path), "r", libver="latest") as f:
        for img_1 in f.keys():
            matches_dict[img_1] = {}
            for img_2 in f[img_1].keys():
                matches_dict[img_1][img_2] = {}
                matches_dict[img_1][img_2]["matches0"] = f[img_1][img_2]["matches0"].__array__()
                matches_dict[img_1][img_2]["matching_scores0"] = f[img_1][img_2]["matching_scores0"].__array__()

    # iterate through all original pairs and create crops
    original_pairs = list(list_h5_names(paths.matches_path))
    for pair in tqdm(original_pairs, desc="Processing pairs...", ncols=80):
        img_1, img_2 = pair.split("/")

        # offsets to transform the keypoints from "crop spaces" to the original image spaces
        offsets = {}

        # get original keypoints and matches
        kp_1 = get_keypoints(paths.features_path, img_1).astype(np.int32)
        kp_2 = get_keypoints(paths.features_path, img_2).astype(np.int32)
        matches, scores = get_matches(paths.matches_path, img_1, img_2)

        if len(matches) < 100:
            continue  # too few matches

        # get top 80% matches
        threshold = np.quantile(scores, 0.2)
        mask = scores >= threshold
        top_matches = matches[mask]

        # compute bounding boxes based on the keypoints of the top 80% matches
        top_kp_1 = kp_1[top_matches[:, 0]]
        top_kp_2 = kp_2[top_matches[:, 1]]
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

        # delete temporary directory with intermediate files from previous matching
        if os.path.exists(paths.cropping_dir):
            shutil.rmtree(paths.cropping_dir)

        # set up new empty temporary directories and save crops
        paths.cropping_dir.mkdir(parents=True, exist_ok=True)
        paths.cropped_image_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(paths.cropped_image_dir / img_1), cropped_image_1)
        cv2.imwrite(str(paths.cropped_image_dir / img_2), cropped_image_2)

        # create new matching pair and save offsets for image space transformations
        offsets[img_1] = (top_kp_1[:, 0].min(), top_kp_1[:, 1].min())
        offsets[img_2] = (top_kp_2[:, 0].min(), top_kp_2[:, 1].min())

        # create text file with the current pair only
        with open(paths.cropped_pairs_path, "w") as f:
            f.write(f"{img_1} {img_2}\n")

        # extract and match features using the cropped images
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

        # transform keypoints from cropped image spaces to original image spaces")
        with h5py.File(str(paths.cropped_features_path), "r+", libver="latest") as f:
            for name in [img_1, img_2]:
                keypoints = f[name]["keypoints"].__array__()
                keypoints[:, 0] += offsets[name][0]
                keypoints[:, 1] += offsets[name][1]
                f[name]["keypoints"][...] = keypoints

        # add new matches to dictionary
        offset = keypoints_dict[img_2].shape[0]  # OLD number of keypoints in SECOND image!
        with h5py.File(str(paths.cropped_matches_path), "r", libver="latest") as f:
            new_matches = f[img_1][img_2]["matches0"].__array__()
            new_matches += offset * np.where(new_matches != -1, 1, 0)
            matches_dict[img_1][img_2]["matches0"] = np.concatenate(
                [matches_dict[img_1][img_2]["matches0"], new_matches], axis=0
            ).astype(np.int32)

            new_scores = f[img_1][img_2]["matching_scores0"].__array__()
            matches_dict[img_1][img_2]["matching_scores0"] = np.concatenate(
                [matches_dict[img_1][img_2]["matching_scores0"], new_scores], axis=0
            )

        # add new keypoints to dictionary
        keypoints_dict[img_1] = np.concatenate(
            [keypoints_dict[img_1], get_keypoints(paths.cropped_features_path, img_1)],
            axis=0,
        )
        keypoints_dict[img_2] = np.concatenate(
            [keypoints_dict[img_2], get_keypoints(paths.cropped_features_path, img_2)],
            axis=0,
        )

    # save final keypoints
    kp_ds = h5py.File(paths.features_path, "w")
    for img in keypoints_dict:
        kp_ds.create_group(img)
        kp_ds[img].create_dataset("keypoints", data=keypoints_dict[img])
    kp_ds.close()

    # save final matches
    matches_ds = h5py.File(paths.matches_path, "w")
    for img1 in matches_dict:
        matches_ds.create_group(img1)
        for img2 in matches_dict[img1].keys():
            matches_ds[img1].create_group(img2)
            matches_ds[img1][img2].create_dataset(
                "matches0", data=matches_dict[img1][img2]["matches0"]
            )
            matches_ds[img1][img2].create_dataset(
                "matching_scores0", data=matches_dict[img1][img2]["matching_scores0"]
            )
    matches_ds.close()
