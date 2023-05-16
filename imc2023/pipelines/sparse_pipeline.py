from hloc import extract_features, match_features

from imc2023.pipelines.pipeline import Pipeline


class SparsePipeline(Pipeline):
    def extract_features(self) -> None:
        """Extract features from the images."""
        self.log_step("Extract features")

        if self.paths.features_path.exists():
            print(f"Features already at {self.paths.features_path}")
            return

        extract_features.main(
            conf=self.config["features"],
            image_dir=self.paths.image_dir,
            image_list=self.img_list,
            feature_path=self.paths.features_path,
        )

    def match_features(self) -> None:
        """Match features between images."""
        self.log_step("Match features")

        if self.paths.matches_path.exists():
            print(f"Matches already at {self.paths.matches_path}")
            return

        match_features.main(
            conf=self.config["matches"],
            pairs=self.paths.pairs_path,
            features=self.paths.features_path,
            matches=self.paths.matches_path,
        )
