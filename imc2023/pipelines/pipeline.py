"""Abstract pipeline class."""
import argparse
import gc
import logging
import shutil
import subprocess
import time
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List

import cv2
import h5py
import numpy as np
import pycolmap
from hloc import (
    extract_features,
    localize_sfm,
    pairs_from_exhaustive,
    pairs_from_retrieval,
    reconstruction,
)
from hloc.utils.database import COLMAPDatabase, blob_to_array
from hloc.utils.io import list_h5_names
from hloc.utils.read_write_model import CAMERA_MODEL_NAMES

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

    def back_rotate_cameras(self):
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

        self.pixsfm = (
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
        if self.pixsfm:
            logging.info("Using PixSfM")

            if not self.paths.cache.exists():
                self.paths.cache.mkdir(parents=True)

            pixsfm_config_name = (
                Path(self.args.pixsfm_config).parent / "low_memory.yaml"
                if len(self.img_list) > self.args.pixsfm_low_mem_threshold
                else self.args.pixsfm_config
            )

            logging.info(f"Using PixSfM config {pixsfm_config_name}")

            if self.args.kaggle:
                python_path = "/kaggle/working/venv/bin/python"
            else:
                python_path = "python"

            proc = subprocess.Popen(
                [
                    python_path,
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
                    str(pixsfm_config_name),
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
            mapper_options.min_num_matches = 10

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
                # skip_geometric_verification=True,
            )

        if self.sparse_model is not None:
            self.sparse_model.write(self.paths.sfm_dir)

        gc.collect()

    def localize_unregistered(self) -> None:
        """Try to localize unregistered images."""
        self.log_step("Localize unregistered images")

        if self.sparse_model is None:
            logging.info("No sparse model reconstructed, skipping localization")
            return

        reg_image_names = [im.name for imid, im in self.sparse_model.images.items()]

        missing = list(set(self.img_list) - set(reg_image_names))
        logging.info(f"Found {len(missing)} unregistered images")

        if len(missing) == 0:
            return

        if self.pixsfm and (self.paths.sfm_dir / "refined_keypoints.h5").exists():
            logging.info("Using PixSfM keypoints")
            features = self.paths.sfm_dir / "refined_keypoints.h5"
            database_path = self.paths.sfm_dir / "hloc" / "database.db"
        else:
            logging.info("Using HLoc keypoints")
            features = self.paths.features_path
            database_path = self.paths.sfm_dir / "database.db"

        db = COLMAPDatabase.connect(database_path)
        for img_name in missing:
            est_conf = pycolmap.AbsolutePoseEstimationOptions()
            refine_conf = pycolmap.AbsolutePoseRefinementOptions()
            ((image_id,),) = db.execute("SELECT image_id FROM images WHERE name=?", (img_name,))
            ((camera_id,),) = db.execute(
                "SELECT camera_id FROM images WHERE image_id=?", (image_id,)
            )

            if camera_id in self.sparse_model.cameras:
                # Just reuse a camera that was already optimized by one of the registered images.
                logging.info(f"Found camera {camera_id} for {img_name} in initial model")
                camera = self.sparse_model.cameras[camera_id]
                logging.info(f"Using camera {camera_id} for {img_name} from initial model")
            else:
                # Infer initial parameters from EXIF and refine them.
                camera = pycolmap.infer_camera_from_image(self.paths.image_dir / img_name)
                camera.camera_id = camera_id
                self.sparse_model.add_camera(camera)
                logging.info(f"Inferring camera for {img_name}")

                est_conf.estimate_focal_length = True
                refine_conf.refine_focal_length = True
                refine_conf.refine_extra_params = True

            conf = {
                "estimation": est_conf.todict(),
                "refinement": refine_conf.todict(),
            }

            logging.info(f"localizing {img_name}")
            q = [(img_name, camera)]

            # localize
            logs = localize_sfm.main(
                self.sparse_model,
                q,
                self.paths.pairs_path,
                features,
                self.paths.matches_path,
                self.paths.scene_dir / "loc.txt",
                covisibility_clustering=True,
                ransac_thresh=10,
                config=conf,
            )

            for q, v in logs["loc"].items():
                v = v["log_clusters"][v["best_cluster"]]
                n_inliers = v["PnP_ret"]["num_inliers"]
                mean_inlier_dist = np.mean(list(v["PnP_ret"]["inliers"]))
                kpts_db = len(v["db"])
                im = pycolmap.Image(
                    q,
                    tvec=v["PnP_ret"]["tvec"],
                    qvec=v["PnP_ret"]["qvec"],
                    id=image_id,
                    camera_id=camera_id,
                )
                im.registered = True
                self.sparse_model.add_image(im)
                logging.info(
                    f"added {q} with {n_inliers} inliers, mean inlier dist {mean_inlier_dist}, {kpts_db} db kpts"
                )

        self.sparse_model.write(self.paths.sfm_dir)

    def run(self) -> None:
        """Run the pipeline."""
        self.timing = {
            "preprocessing": time_function(self.preprocess)(),
            "pairs-extraction": time_function(self.get_pairs)(),
            "feature-extraction": time_function(self.extract_features)(),
            "feature-matching": time_function(self.match_features)(),
            "create-ensemble": time_function(self.create_ensemble)(),
            "rotate-keypoints": time_function(self.rotate_keypoints)(),
            "sfm": time_function(self.sfm)(),
            "back-rotate-cameras": time_function(self.back_rotate_cameras)(),
            "localize-unreg": time_function(self.localize_unregistered)(),
        }
