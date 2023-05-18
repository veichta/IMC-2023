"""Abstract pipeline class."""
import logging
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
        overwrite: bool = False,
    ) -> None:
        """Initialize the pipeline.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
            paths (DataPaths): Data paths.
            img_list (List[str]): List of image names.
            use_pixsfm (bool, optional): Whether to use PixSFM. Defaults to False.
        """
        self.config = config
        self.paths = paths
        self.img_list = img_list
        self.use_pixsfm = use_pixsfm
        self.overwrite = overwrite

        self.sparse_model = None

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
            extract_features.main(
                conf=self.config["retrieval"],
                image_dir=self.paths.image_dir,
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
            self.sparse_model, _ = refiner.run(
                output_dir=self.paths.sfm_dir,
                image_dir=self.paths.image_dir,
                pairs_path=self.paths.pairs_path,
                features_path=self.paths.features_path,
                matches_path=self.paths.matches_path,
                cache_path=self.paths.cache,
                verbose=False,
            )
            # return
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
        self.sfm()
