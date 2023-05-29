"""Abstract pipeline class."""
import argparse
import logging
import shutil
import subprocess
import time
from abc import abstractmethod
from typing import Any, Dict, List

import cv2
import h5py
import numpy as np
import pycolmap
from hloc import extract_features, pairs_from_exhaustive, pairs_from_retrieval, reconstruction
from hloc.utils.io import list_h5_names

from imc2023.cropping import crop_matching
from imc2023.preprocessing import preprocess_image_dir
from imc2023.utils.concatenate import concat_features, concat_matches
from imc2023.utils.utils import DataPaths
from imc2023.utils import rot_mat_z, rotmat2qvec


def time_function(func):
    """Time a function."""

    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        return time.time() - start

    return wrapper


class Pipeline:
    """Abstract pipeline class."""

    def __init__(
        self,
        config: Dict[str, Any],
        paths: DataPaths,
        img_list: List[str],
        args: argparse.Namespace,
    ) -> None:
        """Initialize the pipeline.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
            paths (DataPaths): Data paths.
            img_list (List[str]): List of image names.
            # use_pixsfm (bool, optional): Whether to use PixSFM. Defaults to False.
            # pixsfm_max_imgs (int, optional): Max number of images for PixSFM. Defaults to 9999.
            # pixsfm_config (str, optional): Which PixSFM config to use. Defaults to low_memory.
            # pixsfm_script_path (str, optional): Path to run_pixsfm.py. Needs to be changed for Euler.
            # use_rotation_matching (bool, optional): Whether to use rotation matching. Defaults to False.
            overwrite (bool, optional): Whether to overwrite previous output files. Defaults to False.
        """
        self.config = config
        self.paths = paths
        self.img_list = img_list
        self.use_pixsfm = args.pixsfm
        self.pixsfm_max_imgs = args.pixsfm_max_imgs
        self.pixsfm_config = args.pixsfm_config
        self.pixsfm_script_path = args.pixsfm_script_path
        self.use_rotation_matching = args.rotation_matching
        self.use_rotation_wrapper = args.rotation_wrapper
        self.use_cropping = args.cropping
        self.max_rel_crop_size = args.max_rel_crop_size
        self.min_rel_crop_size = args.min_rel_crop_size
        self.overwrite = args.overwrite
        self.args = args

        self.sparse_model = None
        self.rotated_sparse_model = None

        self.is_ensemble = type(self.config["features"]) == list
        if self.is_ensemble:
            assert len(self.config["features"]) == len(
                self.config["matches"]
            ), "Number of features and matches must be equal for ensemble matching."
            assert (
                len(self.config["features"]) == 2
            ), "Only two features are supported for ensemble matching."

        self.rotation_angles = {}

        self.timing = {
            "preprocess": 0,
            "get_pairs": 0,
            "extract_features": 0,
            "match_features": 0,
            "create_ensemble": 0,
            "rotate_keypoints": 0,
            "sfm": 0,
            "localize_unregistered": 0,
            "back-rotate-cameras":0
        }

    def log_step(self, title: str) -> None:
        """Log a title.

        Args:
            title: The title to log.
        """
        logging.info(f"{'=' * 80}")
        logging.info(title)
        logging.info(f"{'=' * 80}")

    def preprocess(self) -> None:
        """Preprocess the images."""
        self.log_step("Preprocessing")
        self.rotation_angles = preprocess_image_dir(
            input_dir=self.paths.input_dir,
            output_dir=self.paths.scene_dir,
            image_list=self.img_list,
            args=self.args,
        )

    def get_pairs(self) -> None:
        """Get pairs of images to match."""
        self.log_step("Get pairs")

        if len(self.img_list) < self.config["n_retrieval"]:
            pairs_from_exhaustive.main(output=self.paths.pairs_path, image_list=self.img_list)
            return

        if self.paths.pairs_path.exists() and not self.overwrite:
            logging.info(f"Pairs already at {self.paths.pairs_path}")
            return
        else:
            if self.use_rotation_matching or self.use_rotation_wrapper:
                image_dir = self.paths.rotated_image_dir
            else:
                image_dir = self.paths.image_dir

            extract_features.main(
                conf=self.config["retrieval"],
                image_dir=image_dir,
                image_list=self.img_list,
                feature_path=self.paths.features_retrieval,
            )

        pairs_from_retrieval.main(
            descriptors=self.paths.features_retrieval,
            num_matched=self.config["n_retrieval"],
            output=self.paths.pairs_path,
        )

    @abstractmethod
    def extract_features(self) -> None:
        """Extract features from the images."""
        pass

    @abstractmethod
    def match_features(self) -> None:
        """Match features between images."""
        pass

    def create_ensemble(self) -> None:
        """Concatenate features and matches."""
        if not self.is_ensemble:
            return

        self.log_step("Creating ensemble")

        feature_path = self.paths.features_path
        if self.use_rotation_matching:
            feature_path = self.paths.rotated_features_path

        fpath1 = self.paths.features_path.parent / f'{self.config["features"][0]["output"]}.h5'
        fpath2 = self.paths.features_path.parent / f'{self.config["features"][1]["output"]}.h5'

        concat_features(
            features1=fpath1,
            features2=fpath2,
            out_path=feature_path,
        )

        mpath1 = self.paths.matches_path.parent / f'{self.config["matches"][0]["output"]}.h5'
        mpath2 = self.paths.matches_path.parent / f'{self.config["matches"][1]["output"]}.h5'

        concat_matches(
            matches1_path=mpath1,
            matches2_path=mpath2,
            ensemble_features_path=feature_path,
            out_path=self.paths.matches_path,
        )

        pairs = sorted(list(list_h5_names(self.paths.matches_path)))

        with open(self.paths.pairs_path, "w") as f:
            for pair in pairs:
                p = pair.split("/")
                f.write(f"{p[0]} {p[1]}\n")
    
    def match_crops(self):
        """Crop images for each pair and use them to add additional matches."""
        if not self.use_cropping:
            return
        
        self.log_step("Performing crop matching")
        crop_matching(self)

    def back_rotate_cameras(self):
        """Rotate R and t for each rotated camera. """
        if not self.use_rotation_wrapper:
            return
        self.log_step("Back-rotate camera poses")
        for id, im in self.sparse_model.images.items():
            angle = self.rotation_angles[im.name]
            if angle !=0:
                # back rotate <Image 'image_id=30, camera_id=30, name="DSC_6633.JPG", triangulated=404/3133'> by 90
                # logging.info(f"back rotate {im} by {angle}")
                rotmat = rot_mat_z(angle)
                # logging.info(rotmat)
                R = im.rotmat()
                t = np.array(im.tvec)
                self.sparse_model.images[id].tvec = rotmat @t
                self.sparse_model.images[id].qvec = rotmat2qvec(rotmat @ R)
        # self.sparse_model.write(self.paths.sfm_dir)
        # swap the two image folders
        image_dir = self.paths.rotated_image_dir
        self.paths.rotated_image_dir = self.paths.image_dir
        self.paths.image_dir = image_dir

    def rotate_keypoints(self) -> None:
        """Rotate keypoints back after the rotation matching."""
        if not self.use_rotation_matching:
            return

        self.log_step("Rotating keypoints")

        logging.info(f"Using rotated features from {self.paths.rotated_features_path}")
        shutil.copy(self.paths.rotated_features_path, self.paths.features_path)

        logging.info(f"Writing rotated keypoints to {self.paths.features_path}")
        with h5py.File(str(self.paths.features_path), "r+", libver="latest") as f:
            for image_fn, angle in self.rotation_angles.items():
                if angle == 0:
                    continue

                keypoints = f[image_fn]["keypoints"].__array__()
                y_max, x_max = cv2.imread(str(self.paths.rotated_image_dir / image_fn)).shape[:2]

                new_keypoints = np.zeros_like(keypoints)
                if angle == 90:
                    # rotate keypoints by -90 degrees
                    # ==> (x,y) becomes (y, x_max - x)
                    new_keypoints[:, 0] = keypoints[:, 1]
                    new_keypoints[:, 1] = x_max - keypoints[:, 0]-1
                elif angle == 180:
                    # rotate keypoints by 180 degrees
                    # ==> (x,y) becomes (x_max - x, y_max - y)
                    new_keypoints[:, 0] = x_max - keypoints[:, 0]-1
                    new_keypoints[:, 1] = y_max - keypoints[:, 1]-1
                elif angle == 270:
                    # rotate keypoints by +90 degrees
                    # ==> (x,y) becomes (y_max - y, x)
                    new_keypoints[:, 0] = y_max - keypoints[:, 1]-1
                    new_keypoints[:, 1] = keypoints[:, 0]
                f[image_fn]["keypoints"][...] = new_keypoints

    def sfm(self) -> None:
        """Run Structure from Motion."""
        self.log_step("Run SfM")

        if self.paths.sfm_dir.exists() and not self.overwrite:
            try:
                self.sparse_model = pycolmap.Reconstruction(self.paths.sfm_dir)
                logging.info(f"Sparse model already at {self.paths.sfm_dir}")
                return
            except ValueError:
                self.sparse_model = None

        logging.info(f"Using images from {self.paths.image_dir}")
        logging.info(f"Using pairs from {self.paths.pairs_path}")
        logging.info(f"Using features from {self.paths.features_path}")
        logging.info(f"Using matches from {self.paths.matches_path}")

        if self.use_pixsfm and len(self.img_list) <= self.pixsfm_max_imgs:
            logging.info("Using PixSfM")

            if not self.paths.cache.exists():
                self.paths.cache.mkdir(parents=True)

            proc = subprocess.Popen(
                [
                    "python",
                    self.pixsfm_script_path,
                    "--sfm_dir",
                    str(self.paths.sfm_dir),
                    "--image_dir",
                    str(self.paths.image_dir),
                    "--pairs_path",
                    str(self.paths.pairs_path),
                    "--features_path",
                    str(self.paths.features_path),
                    "--matches_path",
                    str(self.paths.matches_path),
                    "--cache_path",
                    str(self.paths.cache),
                    "--pixsfm_config",
                    self.pixsfm_config,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            try:
                logging.info(
                    "Running PixSfM in subprocess (no console output until PixSfM finishes)"
                )
                output, error = proc.communicate()
                logging.info(output.decode())
                logging.error(error.decode())

                # subprocess writes sfm model to disk => load model in main process
                if self.paths.sfm_dir.exists():
                    try:
                        self.sparse_model = pycolmap.Reconstruction(self.paths.sfm_dir)
                    except ValueError:
                        self.sparse_model = None
            except Exception:
                logging.warning("Could not reconstruct model with PixSfM.")
                self.sparse_model = None
        else: 
            self.sparse_model = reconstruction.main(
                sfm_dir=self.paths.sfm_dir,
                image_dir=self.paths.image_dir,
                image_list=self.img_list,
                pairs=self.paths.pairs_path,
                features=self.paths.features_path,
                matches=self.paths.matches_path,
                verbose=False,
            )

        if self.sparse_model is not None:
            self.sparse_model.write(self.paths.sfm_dir)

    def localize_unregistered(self) -> None:
        """Try to localize unregistered images."""
        pass

    def run(self) -> None:
        """Run the pipeline."""
        self.timing = {
            "preprocessing": time_function(self.preprocess)(),
            "pairs-extraction": time_function(self.get_pairs)(),
            "feature-extraction": time_function(self.extract_features)(),
            "feature-matching": time_function(self.match_features)(),
            "create-ensemble": time_function(self.create_ensemble)(),
            "crop-matching": time_function(self.match_crops)(),
            "rotate-keypoints": time_function(self.rotate_keypoints)(),
            "sfm": time_function(self.sfm)(),
            "back-rotate-cameras":time_function(self.back_rotate_cameras)(),
            "localize-unreg": time_function(self.localize_unregistered)(),
        }
