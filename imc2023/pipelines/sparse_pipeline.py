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

        if feature_path.exists() and self.overwrite:
            logging.info(f"Removing {feature_path}")
            feature_path.unlink()

        if self.is_ensemble:
            for config in self.config["features"]:
                extract_features.main(
                    conf=config,
                    image_dir=image_dir,
                    image_list=self.img_list,
                    feature_path=feature_path.parent / f'{config["output"]}.h5',
                )
        else:
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

        if self.paths.matches_path.exists() and self.overwrite:
            logging.info(f"Removing {self.paths.matches_path}")
            self.paths.matches_path.unlink()

        if self.is_ensemble:
            for feat_config, match_config in zip(self.config["features"], self.config["matches"]):
                match_features.main(
                    conf=match_config,
                    pairs=self.paths.pairs_path,
                    features=feature_path.parent / f'{feat_config["output"]}.h5',
                    matches=self.paths.matches_path.parent / f'{match_config["output"]}.h5',
                )
        else:
            match_features.main(
                conf=self.config["matches"],
                pairs=self.paths.pairs_path,
                features=feature_path,
                matches=self.paths.matches_path,
            )
