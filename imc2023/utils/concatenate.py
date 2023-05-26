import logging
from pathlib import Path
from typing import Tuple

import h5py as h5
import numpy as np
from hloc.utils.io import find_pair, list_h5_names
from tqdm import tqdm


def concat_features(features1: Path, features2: Path, out_path: Path) -> None:
    """Concatenate features from two h5 files into one h5 file.

    Args:
        features1 (Path): Path to first h5 file.
        features2 (Path): Path to second h5 file.
        out_path (Path): Path to output h5 file.
    """
    # read features
    img_list = list_h5_names(features1) + list_h5_names(features2)
    img_list = list(set(img_list))
    ensemble_features = {}

    with h5.File(features1, "r") as f1:
        with h5.File(features2, "r") as f2:
            for img in tqdm(img_list, desc="concatenating features", ncols=80):
                kpts1 = f1[img]["keypoints"] if img in f1.keys() else np.array([])
                kpts2 = f2[img]["keypoints"] if img in f2.keys() else np.array([])

                scores1 = f1[img]["scores"] if img in f1.keys() else np.array([])
                scores2 = f2[img]["scores"] if img in f2.keys() else np.array([])

                n_feats1 = len(kpts1) if img in f1.keys() else 0
                n_feats2 = len(kpts2) if img in f2.keys() else 0

                keypoints = np.concatenate([kpts1, kpts2], axis=0)
                scores = np.concatenate([scores1, scores2], axis=0)

                ensemble_features[img] = {
                    "keypoints": keypoints,
                    "scores": scores,
                    "counts": [n_feats1, n_feats2],
                }

    # write features
    ens_kp_ds = h5.File(out_path, "w")
    for img in ensemble_features:
        ens_kp_ds.create_group(img)
        for k in ensemble_features[img].keys():
            ens_kp_ds[img].create_dataset(k, data=ensemble_features[img][k])

    ens_kp_ds.close()


def reverse_matches(
    matches: np.ndarray, scores: np.ndarray, num_kpts1: int, num_kpts2: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Reverse matches between image 1 and image 2.

    Args:
        matches (np.ndarray): Matches from image 1 to image 2.
        scores (np.ndarray): Scores of matches.
        num_kpts1 (int): Number of keypoints in image 1.
        num_kpts2 (int): Number of keypoints in image 2.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Reversed matches and scores.
    """
    rev_matches = np.ones(num_kpts2) * -1
    rev_scores = np.zeros(num_kpts2)

    assert len(matches) == num_kpts1, "Number of matches must equal number of keypoints in image 1"
    assert np.max(matches) < num_kpts2, "Matches must be indices of keypoints in image 2"

    # matches is a list of length nkps1 with each value being either -1 or the index of the match in
    # nkps2
    for i, m in enumerate(matches):
        if m != -1:
            rev_matches[m] = i
            rev_scores[m] = scores[i]

    return rev_matches.astype(int), rev_scores


def extract_matches(
    matches: np.ndarray, features: np.ndarray, name0: str, name1: str, idx=0
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract matches from a pair of images.

    Args:
        matches (np.ndarray): Matches between images.
        features (np.ndarray): Concatenated features of images.
        name0 (str): Name of image 0.
        name1 (str): Name of image 1.
        idx (int, optional): Index of image in concatenated features. Defaults to 0.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Matches and scores.
    """
    nkpts0 = features[name0]["counts"][idx]
    nkpts1 = features[name1]["counts"][idx]

    try:
        p, rev = find_pair(matches, name0, name1)
    except ValueError:
        m = np.ones(nkpts0) * -1
        sc = np.zeros(nkpts0)
        return m, sc

    m = matches[p]["matches0"].__array__()
    sc = matches[p]["matching_scores0"].__array__()

    return reverse_matches(m, sc, nkpts1, nkpts0) if rev else (m, sc)


def concat_matches(
    matches1_path: Path, matches2_path: Path, ensemble_features_path: Path, out_path: Path
):
    # concat matches
    ensemble_matches = {}
    with h5.File(matches1_path, "r") as matches1:
        with h5.File(matches2_path, "r") as matches2:
            with h5.File(ensemble_features_path, "r") as ensemble_features:
                pairs = list_h5_names(matches1_path) + list_h5_names(matches2_path)
                pairs = [sorted(p.split("/"))[0] + "/" + sorted(p.split("/"))[1] for p in pairs]
                pairs = sorted(list(set(pairs)))

                logging.info(f"Found {len(pairs)} unique pairs")
                logging.info(f"Pairs in matches1: {len(list_h5_names(matches1_path))}")
                logging.info(f"Pairs in matches2: {len(list_h5_names(matches2_path))}")

                for pair in tqdm(pairs, desc="concatenating matches", ncols=80):
                    name0, name1 = pair.split("/")

                    # prepare dict
                    if name0 not in ensemble_matches:
                        ensemble_matches[name0] = {}
                    if name1 not in ensemble_matches[name0]:
                        ensemble_matches[name0][name1] = {}

                    # get matches1
                    m1, sc1 = extract_matches(matches1, ensemble_features, name0, name1, idx=0)

                    # get matches2
                    m2, sc2 = extract_matches(matches2, ensemble_features, name0, name1, idx=1)

                    # concat matches
                    offset = ensemble_features[name1]["counts"][0]
                    m2 += offset * np.where(m2 != -1, 1, 0)

                    ensemble_matches[name0][name1]["matches0"] = np.concatenate([m1, m2], axis=0)

                    ensemble_matches[name0][name1]["matching_scores0"] = np.concatenate(
                        [sc1, sc2], axis=0
                    )

    ens_matches_ds = h5.File(out_path, "w")
    for img1 in ensemble_matches:
        ens_matches_ds.create_group(img1)
        for img2 in ensemble_matches[img1].keys():
            ens_matches_ds[img1].create_group(img2)
            for k in ensemble_matches[img1][img2].keys():
                ens_matches_ds[img1][img2].create_dataset(k, data=ensemble_matches[img1][img2][k])

    ens_matches_ds.close()
