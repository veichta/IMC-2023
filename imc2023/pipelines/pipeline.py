"""Abstract pipeline class."""
import logging
import shutil
import h5py
import cv2
import numpy as np
from abc import abstractmethod
from typing import Any, Dict, List

import pycolmap
from hloc import extract_features, pairs_from_exhaustive, pairs_from_retrieval, reconstruction
from pixsfm.refine_hloc import PixSfM

from imc2023.utils.utils import DataPaths


class Pipeline:
    """Abstract pipeline class."""

    def __init__(
        self,
        config: Dict[str, Any],
        paths: DataPaths,
        img_list: List[str],
        use_pixsfm: bool = False,
        use_rotation_matching : bool = False,
        rotation_angles: Dict[str, int] = None,
        overwrite: bool = False,
    ) -> None:
        """Initialize the pipeline.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
            paths (DataPaths): Data paths.
            img_list (List[str]): List of image names.
            use_pixsfm (bool, optional): Whether to use PixSFM. Defaults to False.
            use_rotation_matching (bool, optional): Whether to use rotation matching. Defaults to False.
            rotation_angles (Dict[str, int]): Angles to undo rotation of keypoints. Defaults to None.
            overwrite (bool, optional): Whether to overwrite previous output files. Defaults to False.
        """
        self.config = config
        self.paths = paths
        self.img_list = img_list
        self.use_pixsfm = use_pixsfm
        self.use_rotation_matching = use_rotation_matching
        self.overwrite = overwrite

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

    def rotate_keypoints(self) -> None:
        """Rotate keypoints back after the rotation matching."""
        if not self.use_rotation_matching:
            return
        
        self.log_step("Rotating keypoints")

        shutil.copy(self.paths.rotated_features_path, self.paths.features_path)
        
        with h5py.File(str(self.paths.features_path), 'r+', libver='latest') as f:
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

        if self.use_pixsfm:
            if not self.paths.cache.exists():
                self.paths.cache.mkdir(parents=True)

            refiner = PixSfM(conf=self.config["refinements"])
            try:
                self.sparse_model, _ = refiner.run(
                    output_dir=self.paths.sfm_dir,
                    image_dir=self.paths.image_dir,
                    pairs_path=self.paths.pairs_path,
                    features_path=self.paths.features_path,
                    matches_path=self.paths.matches_path,
                    cache_path=self.paths.cache,
                    verbose=False,
                )
            except ValueError:
                logging.warning("Could not reconstruct model.")
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
        self.rotate_keypoints()
        self.sfm()
