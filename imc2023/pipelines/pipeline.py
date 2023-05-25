"""Abstract pipeline class."""
import logging
import shutil
import subprocess
from abc import abstractmethod
from typing import Any, Dict, List

import cv2
import h5py
import numpy as np
import pycolmap
from hloc import extract_features, pairs_from_exhaustive, pairs_from_retrieval, reconstruction

from imc2023.utils.concatenate import concat_features, concat_matches
from imc2023.utils.utils import DataPaths


class Pipeline:
    """Abstract pipeline class."""

    def __init__(
        self,
        config: Dict[str, Any],
        paths: DataPaths,
        img_list: List[str],
        use_pixsfm: bool = False,
        pixsfm_max_imgs: int = 9999,
        pixsfm_config: str = "low_memory",
        pixsfm_script_path: str = "/kaggle/working/run_pixsfm.py",
        use_rotation_matching: bool = False,
        rotation_angles: Dict[str, int] = None,
        overwrite: bool = False,
    ) -> None:
        """Initialize the pipeline.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
            paths (DataPaths): Data paths.
            img_list (List[str]): List of image names.
            use_pixsfm (bool, optional): Whether to use PixSFM. Defaults to False.
            pixsfm_max_imgs (int, optional): Max number of images for PixSFM. Defaults to 9999.
            pixsfm_config (str, optional): Which PixSFM config to use. Defaults to low_memory.
            pixsfm_script_path (str, optional): Path to run_pixsfm.py. Needs to be changed for Euler.
            use_rotation_matching (bool, optional): Whether to use rotation matching. Defaults to False.
            rotation_angles (Dict[str, int]): Angles to undo rotation of keypoints. Defaults to None.
            overwrite (bool, optional): Whether to overwrite previous output files. Defaults to False.
        """
        self.config = config
        self.paths = paths
        self.img_list = img_list
        self.use_pixsfm = use_pixsfm
        self.pixsfm_max_imgs = pixsfm_max_imgs
        self.pixsfm_config = pixsfm_config
        self.pixsfm_script_path = pixsfm_script_path
        self.use_rotation_matching = use_rotation_matching
        self.overwrite = overwrite

        self.is_ensemble = type(self.config["features"]) == list
        if self.is_ensemble:
            assert len(self.config["features"]) == len(
                self.config["matches"]
            ), "Number of features and matches must be equal for ensemble matching."
            assert (
                len(self.config["features"]) == 2
            ), "Only two features are supported for ensemble matching."

        self.sparse_model = None

        self.rotation_angles = rotation_angles

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
        pass

    def get_pairs(self) -> None:
        """Get pairs of images to match."""
        self.log_step("Get pairs")

        if len(self.img_list) < self.config["n_retrieval"]:
            pairs_from_exhaustive.main(output=self.paths.pairs_path, image_list=self.img_list)
            return

        if self.paths.pairs_path.exists() and not self.overwrite:
            logging.info(f"Pairs already at {self.paths.pairs_path}")
        else:
            if self.use_rotation_matching:
                image_dir = self.paths.rotated_image_dir
            else:
                image_dir = self.paths.image_dir

            extract_features.main(
                conf=self.config["retrieval"],
                image_dir=image_dir,
                image_list=self.img_list,
                feature_path=self.paths.features_retrieval,
            )

        if self.paths.pairs_path.exists() and not self.overwrite:
            logging.info(f"Pairs already at {self.paths.pairs_path}")
            return

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

        fpath1 = self.paths.features_path.parent / self.config["features"][0]["output"]
        fpath2 = self.paths.features_path.parent / self.config["features"][1]["output"]

        concat_features(
            features1=fpath1,
            features2=fpath2,
            out_path=feature_path,
        )

        mpath1 = self.paths.matches_path.parent / self.config["matches"][0]["output"]
        mpath2 = self.paths.matches_path.parent / self.config["matches"][1]["output"]

        concat_matches(
            matches1_path=mpath1,
            matches2_path=mpath2,
            ensemble_features_path=feature_path,
            out_path=self.paths.matches_path,
        )

    def rotate_keypoints(self) -> None:
        """Rotate keypoints back after the rotation matching."""
        if not self.use_rotation_matching:
            return

        self.log_step("Rotating keypoints")

        shutil.copy(self.paths.rotated_features_path, self.paths.features_path)

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
                    new_keypoints[:, 1] = x_max - keypoints[:, 0]
                elif angle == 180:
                    # rotate keypoints by 180 degrees
                    # ==> (x,y) becomes (x_max - x, y_max - y)
                    new_keypoints[:, 0] = x_max - keypoints[:, 0]
                    new_keypoints[:, 1] = y_max - keypoints[:, 1]
                elif angle == 270:
                    # rotate keypoints by +90 degrees
                    # ==> (x,y) becomes (y_max - y, x)
                    new_keypoints[:, 0] = y_max - keypoints[:, 1]
                    new_keypoints[:, 1] = keypoints[:, 0]
                f[image_fn]["keypoints"][...] = new_keypoints

    def sfm(self) -> None:
        """Run Structure from Motion."""
        self.log_step("Run SfM")

        if self.paths.sfm_dir.exists() and not self.overwrite:
            try:
                self.sparse_model = pycolmap.Reconstruction(self.paths.sfm_dir)
                return
            except ValueError:
                self.sparse_model = None

        if self.use_pixsfm and len(self.img_list) <= self.pixsfm_max_imgs:
            logging.info("Using PixSfM")

            if not self.paths.cache.exists():
                self.paths.cache.mkdir(parents=True)

            proc = subprocess.Popen(
                [
                    "python", self.pixsfm_script_path, 
                    "--sfm_dir", str(self.paths.sfm_dir),
                    "--image_dir", str(self.paths.image_dir),
                    "--pairs_path", str(self.paths.pairs_path),
                    "--features_path", str(self.paths.features_path),
                    "--matches_path", str(self.paths.matches_path),
                    "--cache_path", str(self.paths.cache),
                    "--pixsfm_config", self.pixsfm_config,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            try:
                logging.info("Running PixSfM in subprocess (no console output until PixSfM finishes)")
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

    def run(self) -> None:
        """Run the pipeline."""
        self.preprocess()
        self.get_pairs()
        self.extract_features()
        self.match_features()
        self.create_ensemble()
        self.rotate_keypoints()
        self.sfm()
