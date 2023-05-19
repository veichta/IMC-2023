import logging

from hloc import extract_features, match_features

from imc2023.pipelines.pipeline import Pipeline


class SparsePipeline(Pipeline):
    def extract_features(self) -> None:
        """Extract features from the images."""
        self.log_step("Extract features")

        if self.use_rotation_matching:
            feature_path = self.paths.rotated_features_path
            image_dir = self.paths.rotated_image_dir
        else:
            feature_path = self.paths.features_path
            image_dir = self.paths.image_dir

        if feature_path.exists():
            logging.info(f"Features already at {feature_path}")
            return

        extract_features.main(
            conf=self.config["features"],
            image_dir=image_dir,
            image_list=self.img_list,
            feature_path=feature_path,
        )

    def match_features(self) -> None:
        """Match features between images."""
        self.log_step("Match features")

        if self.use_rotation_matching:
            feature_path = self.paths.rotated_features_path
        else:
            feature_path = self.paths.features_path

        if self.paths.matches_path.exists():
            logging.info(f"Matches already at {self.paths.matches_path}")
            return

        match_features.main(
            conf=self.config["matches"],
            pairs=self.paths.pairs_path,
            features=feature_path,
            matches=self.paths.matches_path,
        )
