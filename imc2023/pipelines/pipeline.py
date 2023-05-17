"""Abstract pipeline class."""
import logging
import cv2
import numpy as np
from abc import abstractmethod
from typing import Any, Dict, List

import pycolmap
import dioad.infer
from dioad.processing import rotate_preserve_size
from hloc import extract_features, pairs_from_retrieval, reconstruction

from imc2023.utils.utils import DataPaths

# from pixsfm.refine_hloc import PixSfM


class Pipeline:
    """Abstract pipeline class."""

    def __init__(
        self,
        config: Dict[str, Any],
        paths: DataPaths,
        img_list: List[str],
        use_pixsfm: bool = False,
        use_rotation_matching : bool = False,
        overwrite: bool = False,
    ) -> None:
        """Initialize the pipeline.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
            paths (DataPaths): Data paths.
            img_list (List[str]): List of image names.
            use_pixsfm (bool, optional): Whether to use PixSFM. Defaults to False.
            use_rotation_matching (bool, optional): Whether to use rotation matching. Defaults to False.
            overwrite (bool, optional): Whether to overwrite previous output files. Defaults to False.
        """
        self.config = config
        self.paths = paths
        self.img_list = img_list
        self.use_pixsfm = use_pixsfm
        self.use_rotation_matching = use_rotation_matching
        self.overwrite = overwrite

        self.sparse_model = None

        self.rotation_angles = {}

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

    def rotate_images(self) -> None:
        """Rotate images for rotation matching."""
        if not self.use_rotation_matching:
            return
        
        self.log_step("Rotating images")

        deep_orientation = dioad.infer.Inference()
        for image_fn in self.img_list:
            # predict and save rotation angle
            path = str(self.paths.image_dir / image_fn)
            angle = deep_orientation.predict("vit", path)
            self.rotation_angles[image_fn] = angle

            # rotate and save image
            image = cv2.imread(path)
            image = rotate_preserve_size(path, -angle, (400, 400), False) # TODO: use better image sizes
            cv2.imwrite(str(self.paths.rotated_image_dir / image_fn), np.array(image))

    def get_pairs(self) -> None:
        """Get pairs of images to match."""
        self.log_step("Get pairs")

        if self.paths.pairs_path.exists() and not self.overwrite:
            logging.info(f"Pairs already at {self.paths.pairs_path}")
        else:
            if self.use_rotation_matching:
                image_dir = self.paths.rotated_image_dir
                feature_path = self.paths.rotated_features_path
            else:
                image_dir = self.paths.image_dir
                feature_path = self.paths.features_retrieval

            extract_features.main(
                conf=self.config["retrieval"],
                image_dir=image_dir,
                image_list=self.img_list,
                feature_path=feature_path,
            )

        if self.paths.pairs_path.exists() and not self.overwrite:
            logging.info(f"Pairs already at {self.paths.pairs_path}")
            return

        pairs_from_retrieval.main(
            descriptors=self.paths.features_retrieval,
            num_matched=min(len(self.img_list), self.config["n_retrieval"]),
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

    def unrotate_keypoints(self) -> None:
        """Unrotate keypoints after the rotation matching."""
        if not self.use_rotation_matching:
            return
        
        self.log_step("Unrotating keypoints")
        
        # TODO 

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
            # refiner = PixSfM(conf=self.config["refinements"])
            # self.sparse_model, _ = refiner.run(
            #     output_dir=self.paths.sfm_dir,
            #     image_dir=self.paths.image_dir,
            #     pairs_path=self.paths.pairs_path,
            #     features_path=self.paths.features_path,
            #     matches_path=self.paths.matches_path,
            #     cache_path=self.paths.cache,
            #     verbose=False,
            # )
            return
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
        self.rotate_images()
        self.get_pairs()
        self.extract_features()
        self.match_features()
        self.unrotate_keypoints()
        self.sfm()
