import sys

sys.path.append("ext_deps/Hierarchical-Localization")
sys.path.append("ext_deps/dioad")

# sys.path.append("/kaggle/input/imc-23-repo/IMC-2023/ext_deps/Hierarchical-Localization")
# sys.path.append("/kaggle/input/imc-23-repo/IMC-2023/ext_deps/dioad")

import argparse
import gc
import json
import logging
import os
import pickle
from pathlib import Path

import cv2
import dioad.infer
import numpy as np
from numba import cuda
from tqdm import tqdm

from imc2023.configs import configs
from imc2023.utils.eval import eval
from imc2023.utils.utils import DataPaths, create_submission, get_data_from_dict, get_data_from_dir

# PARSE ARGS
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True, help="imc dataset")
parser.add_argument("--config", type=str, required=True, choices=configs.keys(), help="config name")
parser.add_argument(
    "--mode", type=str, required=True, choices=["train", "test"], help="train or test"
)
parser.add_argument("--output", type=str, default="outputs", help="output dir")
parser.add_argument("--pixsfm", action="store_true", help="use pixsfm")
parser.add_argument("--pixsfm_max_imgs", type=int, default=9999, help="max number of images for PixSfM")
parser.add_argument("--pixsfm_config", type=str, default="low_memory", help="PixSfM config")
parser.add_argument("--rotation_matching", action="store_true", help="use rotation matching")
parser.add_argument("--overwrite", action="store_true", help="overwrite existing results")
args = parser.parse_args()

# os.makedirs("/kaggle/temp", exist_ok=True)

# args = {
#     "data": "/kaggle/input/image-matching-challenge-2023",
#     "config": "DISK+LG",
#     "mode": "train",
#     "output": "/kaggle/temp",
#     "pixsfm": False,
#     "pixsfm_max_imgs": 9999,
#     "pixsfm_config": "low_memory",
#     "rotation_matching": False,
#     "overwrite": False,
# }

# args = argparse.Namespace(**args)


formatter = logging.Formatter(
    fmt="[%(asctime)s %(name)s %(levelname)s] %(message)s", datefmt="%Y/%m/%d %H:%M:%S"
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False

# SETTINGS
MODE = args.mode  # "train" or "test"
CONF_NAME = args.config
PIXSFM = args.pixsfm
PIXSFM_MAX_IMGS = args.pixsfm_max_imgs
PIXSFM_CONFIG = args.pixsfm_config
ROTATION_MATCHING = args.rotation_matching
OVERWRITE = args.overwrite

# PATHS
data_dir = Path(args.data)

output_dir = f"{args.output}/{CONF_NAME}"
if ROTATION_MATCHING:
    output_dir += "-rot"
if PIXSFM:
    output_dir += "-pixsfm"
output_dir = Path(output_dir)

output_dir.mkdir(exist_ok=True, parents=True)

metrics_path = output_dir / "metrics.pickle"
results_path = output_dir / "results.pickle"
submission_csv_path = Path(f"{output_dir}/submission.csv")

# CONFIG
config = configs[CONF_NAME]
with open(str(output_dir / "config.json"), "w") as jf:
    json.dump(config, jf, indent=4)

logging.info("CONFIG:")
for step, conf in config.items():
    if step == "n_retrieval":
        logging.info(f"  {step}: {conf}")
        continue

    if conf is None:
        logging.info(f"  {step}: None")
        continue

    if type(conf) == list:
        logging.info(f"  {step}:")
        for i, c in enumerate(conf):
            logging.info(f"    {i}:")
            for k, v in c.items():
                logging.info(f"      {k}: {v}")
        continue

    logging.info(f"{step}:")
    for k, v in conf.items():
        logging.info(f"  {k}: {v}")


# SETUP DATA DICT
data_dict = get_data_from_dict(data_dir) if MODE == "test" else get_data_from_dir(data_dir, MODE)

logging.info("DATA:")
for ds, ds_vals in data_dict.items():
    logging.info(ds)
    for scene in ds_vals.keys():
        logging.info(f"  {scene}: {len(data_dict[ds][scene])} imgs")

# ROTATE IMAGES
# the dioad model is huge and barely fits into the GPU
# ==> much more efficient and less crash-prone to load
# the model only once and process all images in the beginning
rotation_angles = {}  # to undo rotations for the keypoints

if ROTATION_MATCHING:
    import tensorflow as tf

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    logging.info("Rotating images for rotation matching:")

    deep_orientation = dioad.infer.Inference(
        load_model_path="ext_deps/dioad/weights/model-vit-ang-loss.h5"
    )
    for dataset in data_dict:
        # SKIP PHOTOTOURISM FOR TRAINING
        if MODE == "train" and dataset == "phototourism":
            continue

        rotation_angles[dataset] = {}
        for scene in data_dict[dataset]:
            logging.info(f"  {dataset} - {scene}")

            rotation_angles[dataset][scene] = {}

            paths = DataPaths(
                data_dir=data_dir,
                output_dir=output_dir,
                dataset=dataset,
                scene=scene,
                mode=MODE,
            )

            if not paths.image_dir.exists():
                continue

            img_list = [Path(p).name for p in data_dict[dataset][scene]]
            n_rotated = 0
            for image_fn in tqdm(img_list, desc=f"Rotating {dataset}/{scene}", ncols=80):
                # predict rotation angle
                path = str(paths.image_dir / image_fn)

                try:
                    angle = deep_orientation.predict("vit", path)
                except:
                    logging.warning(f"Could not predict rotation for {dataset}/{scene}/{image_fn}")
                    angle = np.random.choice([0, 90, 180, 270])

                # round angle to closest multiple of 90° and save it for later
                if angle < 0.0:
                    angle += 360
                angle = (
                    round(angle / 90.0) * 90
                ) % 360  # angle is now an integer in [0, 90, 180, 270]
                rotation_angles[dataset][scene][image_fn] = angle

                if angle != 0:
                    n_rotated += 1

                # rotate and save image
                image = cv2.imread(path)
                if angle == 90:
                    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                elif angle == 180:
                    image = cv2.rotate(image, cv2.ROTATE_180)
                elif angle == 270:
                    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                cv2.imwrite(str(paths.rotated_image_dir / image_fn), image)

            logging.info(f"  {n_rotated} / {len(img_list)} images rotated")

    # free cuda memory
    del deep_orientation
    gc.collect()
    device = cuda.get_current_device()
    device.reset()

# RUN
metrics = {}
out_results = {}

for dataset in data_dict:
    # SKIP PHOTOTOURISM FOR TRAINING
    if MODE == "train" and dataset == "phototourism":
        continue

    logging.info(dataset)

    if dataset not in metrics:
        metrics[dataset] = {}

    if dataset not in out_results:
        out_results[dataset] = {}

    scenes = data_dict[dataset].keys()
    for scene in data_dict[dataset]:
        logging.info(f"{dataset} - {scene}")

        # SETUP PATHS
        paths = DataPaths(
            data_dir=data_dir,
            output_dir=output_dir,
            dataset=dataset,
            scene=scene,
            mode=MODE,
        )

        if not paths.image_dir.exists():
            logging.info(f"Skipping {dataset} - {scene} (no images)")
            continue

        img_list = [Path(p).name for p in data_dict[dataset][scene]]
        out_results[dataset][scene] = {}

        if config["features"] is not None:
            from imc2023.pipelines.sparse_pipeline import SparsePipeline as Pipeline
        else:
            from imc2023.pipelines.dense_pipeline import DensePipeline as Pipeline

        pipeline = Pipeline(
            config=config,
            paths=paths,
            img_list=img_list,
            use_pixsfm=PIXSFM,
            pixsfm_max_imgs=PIXSFM_MAX_IMGS,
            pixsfm_config=PIXSFM_CONFIG,
            use_rotation_matching=ROTATION_MATCHING,
            rotation_angles=rotation_angles[dataset][scene] if ROTATION_MATCHING else None,
            overwrite=OVERWRITE,
        )

        pipeline.run()

        sparse_model = pipeline.sparse_model

        if sparse_model is None:
            continue

        metrics[dataset][scene] = {
            "n_images": len(img_list),
            "n_reg_images": sparse_model.num_reg_images(),
        }

        logging.info(f"Extracting results for {dataset} - {scene}")
        for _, im in sparse_model.images.items():
            img_name = os.path.join(dataset, scene, "images", im.name)
            # problem: tvec is a reference! --> force copy
            out_results[dataset][scene][img_name] = {"R": im.rotmat(), "t": np.array(im.tvec)}

        logging.info("Done...")

        with open(metrics_path, "wb") as handle:
            pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(results_path, "wb") as handle:
            pickle.dump(out_results, handle, protocol=pickle.HIGHEST_PROTOCOL)


create_submission(out_results, data_dict, submission_csv_path)
create_submission(out_results, data_dict, "submission.csv")


if MODE == "train":
    # # WRITE SUBMISSION
    with open(metrics_path, "rb") as handle:
        metrics = pickle.load(handle)
    with open(results_path, "rb") as handle:
        out_results = pickle.load(handle)

    for dataset in metrics:
        logging.info(dataset)
        for scene in metrics[dataset]:
            logging.info(
                f"\t{scene}: {metrics[dataset][scene]['n_reg_images']} / {metrics[dataset][scene]['n_images']}"
            )

    # EVALUATE
    eval(submission_csv="submission.csv", data_dir=data_dir)
