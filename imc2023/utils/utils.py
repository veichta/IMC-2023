import os
from pathlib import Path
from typing import Any, Dict

import numpy as np


def arr_to_str(a):
    return ";".join([str(x) for x in a.reshape(-1)])


def get_data_from_dict(data_dir: str) -> Dict[str, Any]:
    """Function to get data from a dictionary.

    Args:
        data_dir (str): Path to data directory.

    Returns:
        Dict[str, Any]: Description of returned object.
    """
    data_dict = {}
    with open(os.path.join(data_dir, "sample_submission.csv"), "r") as f:
        for i, l in enumerate(f):
            # Skip header.
            if l and i > 0:
                image, dataset, scene, _, _ = l.strip().split(",")
                if dataset not in data_dict:
                    data_dict[dataset] = {}
                if scene not in data_dict[dataset]:
                    data_dict[dataset][scene] = []
                data_dict[dataset][scene].append(image)
    return data_dict


def get_data_from_dir(data_dir: str, mode: str) -> Dict[str, Any]:
    """Function to get data from a directory.

    Args:
        data_dir (str): Path to data directory.
        mode (str): Mode (train or test).

    Raises:
        ValueError: Invalid mode.

    Returns:
        Dict[str, Any]: Data dictionary.
    """
    if mode not in {"train", "test"}:
        raise ValueError(f"Invalid mode: {mode}")

    data_dict = {}
    datasets = [
        x
        for x in os.listdir(os.path.join(data_dir, mode))
        if os.path.isdir(os.path.join(data_dir, mode, x))
    ]
    for dataset in datasets:
        if dataset not in data_dict:
            data_dict[dataset] = {}

        # dataset_dir = f"{DIR}/{MODE}/{dataset}"
        dataset_dir = os.path.join(data_dir, mode, dataset)
        scenes = [x for x in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, x))]
        for scene in scenes:
            image_dir = os.path.join(dataset_dir, scene, "images")
            data_dict[dataset][scene] = []
            for img in os.listdir(image_dir):
                data_dict[dataset][scene].append(os.path.join(dataset, scene, "images", img))

    return data_dict


def create_submission(out_results: Dict[str, Any], data_dict: Dict[str, Any], fname: str):
    """Function to create a submission file.

    Args:
        out_results (Dict[str, Any]): Estimated poses.
        data_dict (Dict[str, Any]): Data dictionary.
        fname (str): Output file name.
    """
    with open(fname, "w") as f:
        f.write("image_path,dataset,scene,rotation_matrix,translation_vector\n")
        for dataset in data_dict:
            res = out_results.get(dataset, {})
            for scene in data_dict[dataset]:
                scene_res = res[scene] if scene in res else {"R": {}, "t": {}}
                for image in data_dict[dataset][scene]:
                    if image in scene_res:
                        print(image)
                        R = np.array(scene_res[image]["R"]).reshape(-1)
                        T = np.array(scene_res[image]["t"]).reshape(-1)
                    else:
                        R = np.eye(3).reshape(-1)
                        T = np.zeros((3))
                    f.write(f"{image},{dataset},{scene},{arr_to_str(R)},{arr_to_str(T)}\n")
    f.close()


class DataPaths:
    def __init__(self, data_dir: str, output_dir: str, dataset: str, scene: str, mode: str):
        """Class to store paths.

        Args:
            data_dir (str): Path to data directory.
            output_dir (str): Path to output directory.
            dataset (str): Dataset name.
            scene (str): Scene name.
            mode (str): Mode (train or test).
        """
        if mode not in {"train", "test"}:
            raise ValueError(f"Invalid mode: {mode}")

        self.image_dir = Path(f"{data_dir}/{mode}/{dataset}/{scene}/images")
        self.rotated_image_dir = Path(f"{data_dir}/{mode}/{dataset}/{scene}/images_rotated")
        self.scene_dir = output_dir / dataset / scene
        self.scene_dir.mkdir(parents=True, exist_ok=True)

        self.sfm_dir = self.scene_dir / "sparse"
        self.pairs_path = self.scene_dir / "pairs.txt"
        self.features_retrieval = self.scene_dir / "features_retrieval.h5"
        self.features_path = self.scene_dir / "features.h5"
        self.rotated_features_path = self.scene_dir / "features_rotated.h5"
        self.matches_path = self.scene_dir / "matches.h5"

        # TODO: Update this.
        self.cache = self.scene_dir / "cache"
