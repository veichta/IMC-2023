"""Abstract pipeline class."""
import argparse
import gc
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
from hloc import (
    extract_features,
    match_features,
    pairs_from_exhaustive,
    pairs_from_retrieval,
    reconstruction,
)
from hloc.utils.io import find_pair, get_keypoints, get_matches, list_h5_names
from tqdm import tqdm

from imc2023.preprocessing import preprocess_image_dir
from imc2023.utils import rot_mat_z, rotmat2qvec
from imc2023.utils.concatenate import concat_features, concat_matches
from imc2023.utils.utils import DataPaths


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
        self.same_shapes = False
        self.args = args

        self.sparse_model = None
        self.rotated_sparse_model = None

        self.is_ensemble = len(self.config["features"]) > 1
        if self.is_ensemble:
            assert len(self.config["features"]) == len(
                self.config["matches"]
            ), "Number of features and matches must be equal for ensemble matching."

        self.rotation_angles = {}
        self.n_rotated = 0

        self.timing = {
            "preprocess": 0,
            "get_pairs": 0,
            "extract_features": 0,
            "match_features": 0,
            "create_ensemble": 0,
            "rotate_keypoints": 0,
            "sfm": 0,
            "localize_unregistered": 0,
            "back-rotate-cameras": 0,
        }

        # log data paths
        logging.info("Data paths:")
        for key, value in self.paths.__dict__.items():
            logging.info(f"  {key}: {value}")

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

        self.rotation_angles, self.same_shapes = preprocess_image_dir(
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
        self.log_step("Creating ensemble")

        if not self.is_ensemble:
            logging.info("Not using ensemble matching")
            return

        feature_path = self.paths.features_path
        if self.use_rotation_matching:
            feature_path = self.paths.rotated_features_path

        # copy first feature and matches to final output
        shutil.copyfile(
            self.paths.features_path.parent / f'{self.config["features"][0]["output"]}.h5',
            feature_path,
        )
        shutil.copyfile(
            self.paths.matches_path.parent / f'{self.config["matches"][0]["output"]}.h5',
            self.paths.matches_path,
        )

        # concatenate features and matches for remaining features and matches
        for i in range(1, len(self.config["features"])):
            feat_path = (
                self.paths.features_path.parent / f'{self.config["features"][i]["output"]}.h5'
            )
            match_path = (
                self.paths.matches_path.parent / f'{self.config["matches"][i]["output"]}.h5'
            )

            concat_features(
                features1=feature_path,
                features2=feat_path,
                out_path=feature_path,
            )

            concat_matches(
                matches1_path=self.paths.matches_path,
                matches2_path=match_path,
                ensemble_features_path=feature_path,
                out_path=self.paths.matches_path,
            )

        # write pairs file
        # TODO: check if this is necessary
        pairs = sorted(list(list_h5_names(self.paths.matches_path)))
        with open(self.paths.pairs_path, "w") as f:
            for pair in pairs:
                p = pair.split("/")
                f.write(f"{p[0]} {p[1]}\n")

    def perform_cropping(self):
        """Crop images for each pair and use them to add additional matches."""
        if not self.use_cropping:
            return

        self.log_step("Performing image cropping")

        logging.info("Creating crops for all matches")

        if self.use_rotation_matching:
            feature_path = self.paths.rotated_features_path
            image_dir = self.paths.rotated_image_dir
        elif self.use_rotation_wrapper:
            feature_path = self.paths.features_path
            # swap the two image folders
            image_dir = self.paths.rotated_image_dir
            self.paths.rotated_image_dir = self.paths.image_dir
            self.paths.image_dir = image_dir
        else:
            feature_path = self.paths.features_path
            image_dir = self.paths.image_dir

        # new list of pairs for the matching on crops
        crop_pairs = []

        # dictionary of offsets to transform the keypoints from "crop spaces" to the original image spaces
        offsets = {}

        # iterate through all original pairs and create crops
        original_pairs = list(list_h5_names(self.paths.matches_path))
        for pair in tqdm(original_pairs, desc="Creating crops", ncols=80):
            img_1, img_2 = pair.split("/")

            # get original keypoints and matches
            kp_1 = get_keypoints(feature_path, img_1).astype(np.int32)
            kp_2 = get_keypoints(feature_path, img_2).astype(np.int32)
            matches, scores = get_matches(self.paths.matches_path, img_1, img_2)

            if len(matches) < 100:
                continue  # too few matches

            # get top 80% matches
            threshold = np.quantile(scores, 0.2)
            mask = scores >= threshold
            top_matches = matches[mask]

            # compute bounding boxes based on the keypoints of the top 80% matches
            top_kp_1 = kp_1[top_matches[:, 0]]
            top_kp_2 = kp_2[top_matches[:, 1]]
            original_image_1 = cv2.imread(str(self.paths.image_dir / img_1))
            original_image_2 = cv2.imread(str(self.paths.image_dir / img_2))
            cropped_image_1 = original_image_1[
                top_kp_1[:, 1].min() : top_kp_1[:, 1].max() + 1,
                top_kp_1[:, 0].min() : top_kp_1[:, 0].max() + 1,
            ]
            cropped_image_2 = original_image_2[
                top_kp_2[:, 1].min() : top_kp_2[:, 1].max() + 1,
                top_kp_2[:, 0].min() : top_kp_2[:, 0].max() + 1,
            ]

            # check if the relative size conditions are fulfilled
            rel_size_1 = cropped_image_1.size / original_image_1.size
            rel_size_2 = cropped_image_2.size / original_image_2.size

            if rel_size_1 <= self.min_rel_crop_size or rel_size_2 < self.min_rel_crop_size:
                # one of the crops or both crops are too small ==> avoid degenerate crops
                continue

            if rel_size_1 >= self.max_rel_crop_size and rel_size_2 >= self.max_rel_crop_size:
                # both crops are almost the same size as the original images
                # ==> crops are not useful (almost same matches as on the original images)
                continue

            # define new names for the crops based on the current pair because each
            # original image will be cropped in a different way for each original match
            name_1 = f"{img_1}#{img_2}#1.jpg"
            name_2 = f"{img_1}#{img_2}#2.jpg"

            # save crops
            cv2.imwrite(str(self.paths.cropped_image_dir / name_1), cropped_image_1)
            cv2.imwrite(str(self.paths.cropped_image_dir / name_2), cropped_image_2)

            # create new matching pair and save offsets for image space transformations
            crop_pairs.append((name_1, name_2))
            offsets[name_1] = (top_kp_1[:, 0].min(), top_kp_1[:, 1].min())
            offsets[name_2] = (top_kp_2[:, 0].min(), top_kp_2[:, 1].min())

        # save new list of crop pairs
        with open(self.paths.cropped_pairs_path, "w") as f:
            for p1, p2 in crop_pairs:
                f.write(f"{p1} {p2}\n")

        logging.info(f"Created {len(crop_pairs)} crop pairs")

        self.offsets = offsets

    def extract_and_match_features_crop(self):
        if not self.use_cropping:
            return

        self.log_step("Extract and match features on crops")

        if self.use_rotation_matching:
            feature_path = self.paths.rotated_features_path
            image_dir = self.paths.rotated_image_dir
        elif self.use_rotation_wrapper:
            feature_path = self.paths.features_path
            # swap the two image folders
            image_dir = self.paths.rotated_image_dir
            self.paths.rotated_image_dir = self.paths.image_dir
            self.paths.image_dir = image_dir
        else:
            feature_path = self.paths.features_path
            image_dir = self.paths.image_dir

        with open(self.paths.cropped_pairs_path, "r") as f:
            crop_pairs = [line.strip().split() for line in f.readlines()]

        # get hloc logger
        hloc_logger = logging.getLogger("hloc")
        hloc_logger.setLevel(logging.WARNING)

        for p in crop_pairs:
            # extract features and match on crop pair
            logging.info(f"Extracting and matching features on crop pair {p}")
            feat_crop_path = self.paths.cropped_features_dir / f"{p[0]}_{p[1]}.h5"
            match_crop_path = self.paths.cropped_matches_dir / f"{p[0]}_{p[1]}.h5"

            # del if exists
            if feat_crop_path.exists():
                feat_crop_path.unlink()
            if match_crop_path.exists():
                match_crop_path.unlink()

            extract_features.main(
                conf=self.config["features"][0],
                image_dir=self.paths.cropped_image_dir,
                image_list=p,
                feature_path=feat_crop_path,
            )

            with open(self.paths.cropped_features_dir / f"tmp_pairs.txt", "w") as f:
                f.write(f"{p[0]} {p[1]}\n")

            match_features.main(
                conf=self.config["matches"][0],
                pairs=self.paths.cropped_features_dir / f"tmp_pairs.txt",
                features=self.paths.cropped_features_dir / f"{p[0]}_{p[1]}.h5",
                matches=self.paths.cropped_matches_dir / f"{p[0]}_{p[1]}.h5",
            )

            # transform keypoints from cropped image spaces to original image spaces
            logging.info(
                f"Transforming keypoints from cropped image spaces to original image spaces for crop pair {p}"
            )
            names = set(self.offsets.keys()) & set(
                list_h5_names(self.paths.cropped_features_dir / f"{p[0]}_{p[1]}.h5")
            )
            logging.info(f"Transforming keypoints for {names}")
            with h5py.File(str(feat_crop_path), "r+", libver="latest") as f:
                for name in names:
                    idx = int(name.split("#")[-1].split(".")[0]) - 1
                    orig_name = name.split("#")[idx]
                    keypoints = f[name]["keypoints"].__array__()
                    keypoints[:, 0] += self.offsets[name][0]
                    keypoints[:, 1] += self.offsets[name][1]

                    scores = f[name]["scores"].__array__()

                    # add keypoints to original image
                    f.create_group(orig_name)
                    f[orig_name].create_dataset("keypoints", data=keypoints)
                    f[orig_name].create_dataset("scores", data=scores)

                    # delete keypoints from crop
                    del f[name]

            # rename matches
            with h5py.File(str(match_crop_path), "r+", libver="latest") as f:
                name0, name1 = p[0], p[1]
                p_, rev = find_pair(f, name0, name1)

                matches = f[p_]["matches0"].__array__()
                scores = f[p_]["matching_scores0"].__array__()

                idx = int(name0.split("#")[-1].split(".")[0]) - 1
                orig_name0 = name0.split("#")[idx]

                idx = int(name1.split("#")[-1].split(".")[0]) - 1
                orig_name1 = name1.split("#")[idx]

                if not rev:
                    f.create_group(orig_name0)
                    f[orig_name0].create_group(orig_name1)
                    f[orig_name0][orig_name1].create_dataset("matches0", data=matches)
                    f[orig_name0][orig_name1].create_dataset("matching_scores0", data=scores)
                else:
                    f.create_group(orig_name1)
                    f[orig_name1].create_group(orig_name0)
                    f[orig_name1][orig_name0].create_dataset("matches1", data=matches)
                    f[orig_name1][orig_name0].create_dataset("matching_scores0", data=scores)

                del f[p_]

            # concatenate features and matches from crops with original features and matches
            logging.info(
                f"Concatenating features and matches from crops with original features and matches for crop pair {p}"
            )
            concat_features(
                feature_path,
                self.paths.cropped_features_dir / f"{p[0]}_{p[1]}.h5",
                feature_path,
            )
            concat_matches(
                self.paths.matches_path,
                match_crop_path,
                feature_path,
                self.paths.matches_path,
            )

        hloc_logger.setLevel(logging.INFO)

    def back_rotate_cameras(self):
        """Rotate R and t for each rotated camera."""
        """Rotate R and t for each rotated camera."""
        self.log_step("Back-rotate camera poses")
        if not self.use_rotation_wrapper:
            logging.info("Not using rotation wrapper")
            return

        if self.sparse_model is None:
            logging.info("No sparse model reconstructed, skipping back-rotation")
            return

        for id, im in self.sparse_model.images.items():
            angle = self.rotation_angles[im.name]
            if angle != 0:
                # back rotate <Image 'image_id=30, camera_id=30, name="DSC_6633.JPG", triangulated=404/3133'> by 90
                # logging.info(f"back rotate {im} by {angle}")
                rotmat = rot_mat_z(angle)
                # logging.info(rotmat)
                R = im.rotmat()
                t = np.array(im.tvec)
                self.sparse_model.images[id].tvec = rotmat @ t
                self.sparse_model.images[id].qvec = rotmat2qvec(rotmat @ R)
        # self.sparse_model.write(self.paths.sfm_dir)

        # swap the two image folders
        # image_dir = self.paths.rotated_image_dir
        # self.paths.rotated_image_dir = self.paths.image_dir
        # self.paths.image_dir = image_dir

    def rotate_keypoints(self) -> None:
        """Rotate keypoints back after the rotation matching."""
        self.log_step("Rotating keypoints")

        if not self.use_rotation_matching:
            logging.info("Not using rotation matching")
            return

        logging.info(f"Using rotated features from {self.paths.rotated_features_path}")
        shutil.copy(self.paths.rotated_features_path, self.paths.features_path)

        logging.info(f"Writing rotated keypoints to {self.paths.features_path}")
        with h5py.File(str(self.paths.features_path), "r+", libver="latest") as f:
            for image_fn, angle in self.rotation_angles.items():
                if angle == 0:
                    continue

                self.n_rotated += 1

                keypoints = f[image_fn]["keypoints"].__array__()
                y_max, x_max = cv2.imread(str(self.paths.rotated_image_dir / image_fn)).shape[:2]

                new_keypoints = np.zeros_like(keypoints)
                if angle == 90:
                    # rotate keypoints by -90 degrees
                    # ==> (x,y) becomes (y, x_max - x)
                    new_keypoints[:, 0] = keypoints[:, 1]
                    new_keypoints[:, 1] = x_max - keypoints[:, 0] - 1
                elif angle == 180:
                    # rotate keypoints by 180 degrees
                    # ==> (x,y) becomes (x_max - x, y_max - y)
                    new_keypoints[:, 0] = x_max - keypoints[:, 0] - 1
                    new_keypoints[:, 1] = y_max - keypoints[:, 1] - 1
                elif angle == 270:
                    # rotate keypoints by +90 degrees
                    # ==> (x,y) becomes (y_max - y, x)
                    new_keypoints[:, 0] = y_max - keypoints[:, 1] - 1
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

        # read images from rotated image dir if rotation wrapper is used
        image_dir = (
            self.paths.rotated_image_dir if self.use_rotation_wrapper else self.paths.image_dir
        )

        camera_mode = pycolmap.CameraMode.AUTO
        if self.same_shapes and self.args.shared_camera:
            camera_mode = pycolmap.CameraMode.SINGLE

        pixsfm = (
            self.use_pixsfm
            and len(self.img_list) <= self.pixsfm_max_imgs
            and (self.n_rotated == 0 or self.args.rotation_wrapper)
        )

        if self.n_rotated != 0 and self.use_pixsfm and not self.args.rotation_wrapper:
            logging.info(f"Not using pixsfm because {self.n_rotated} rotated images are detected")

        logging.info(f"Using images from {image_dir}")
        logging.info(f"Using pairs from {self.paths.pairs_path}")
        logging.info(f"Using features from {self.paths.features_path}")
        logging.info(f"Using matches from {self.paths.matches_path}")
        logging.info(f"Using {camera_mode}")

        gc.collect()
        if pixsfm:
            logging.info("Using PixSfM")

            if not self.paths.cache.exists():
                self.paths.cache.mkdir(parents=True)

            pixsfm_config_name = (
                "low_memory"
                if len(self.img_list) > self.args.pixsfm_low_mem_threshold
                else self.args.pixsfm_config
            )

            logging.info(f"Using PixSfM config {pixsfm_config_name}")

            proc = subprocess.Popen(
                [
                    "python",
                    self.pixsfm_script_path,
                    "--sfm_dir",
                    str(self.paths.sfm_dir),
                    "--image_dir",
                    str(image_dir),
                    "--pairs_path",
                    str(self.paths.pairs_path),
                    "--features_path",
                    str(self.paths.features_path),
                    "--matches_path",
                    str(self.paths.matches_path),
                    "--cache_path",
                    str(self.paths.cache),
                    "--pixsfm_config",
                    pixsfm_config_name,
                    "--camera_mode",
                    "auto" if camera_mode == pycolmap.CameraMode.AUTO else "single",
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
                        logging.warning(
                            f"Could not reconstruct / read model from {self.paths.sfm_dir}."
                        )
                        self.sparse_model = None
            except Exception:
                logging.warning("Could not reconstruct model with PixSfM.")
                self.sparse_model = None
        else:
            mapper_options = pycolmap.IncrementalMapperOptions()
            mapper_options.min_model_size = 6

            self.sparse_model = reconstruction.main(
                sfm_dir=self.paths.sfm_dir,
                image_dir=image_dir,
                image_list=self.img_list,
                pairs=self.paths.pairs_path,
                features=self.paths.features_path,
                matches=self.paths.matches_path,
                camera_mode=camera_mode,
                verbose=False,
                reference_model=self.paths.reference_model,
                mapper_options=mapper_options.todict(),
            )

        if self.sparse_model is not None:
            self.sparse_model.write(self.paths.sfm_dir)

        gc.collect()

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
            "image-cropping": time_function(self.perform_cropping)(),
            "features-and-matches-crop": time_function(self.extract_and_match_features_crop)(),
            "rotate-keypoints": time_function(self.rotate_keypoints)(),
            "sfm": time_function(self.sfm)(),
            "back-rotate-cameras": time_function(self.back_rotate_cameras)(),
            "localize-unreg": time_function(self.localize_unregistered)(),
        }
