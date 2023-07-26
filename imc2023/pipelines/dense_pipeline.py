import logging

from hloc import match_dense

from imc2023.pipelines.pipeline import Pipeline


class DensePipeline(Pipeline):
    def extract_features(self) -> None:
        """Extract features from the images."""
        pass

    def match_features(self) -> None:
        """Match features between images."""
        self.log_step("Match features")

        if self.paths.matches_path.exists():
            logging.info(f"Matches already at {self.paths.matches_path}")
            return

        if self.use_rotation_matching:
            feature_path = self.paths.rotated_features_path
            image_dir = self.paths.rotated_image_dir
        else:
            feature_path = self.paths.features_path
            image_dir = self.paths.image_dir

        match_dense.main(
            conf=self.config["matches"][0],
            image_dir=image_dir,
            pairs=self.paths.pairs_path,
            features=feature_path,
            matches=self.paths.matches_path,
        )
