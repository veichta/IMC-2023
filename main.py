import sys

sys.path.append("ext_deps/Hierarchical-Localization")

import argparse
import json
import logging
import os
import pickle
from pathlib import Path

import numpy as np

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
parser.add_argument("--rotation_matching", action="store_true", help="use rotation matching")
parser.add_argument("--overwrite", action="store_true", help="overwrite existing results")
args = parser.parse_args()

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
ROTATION_MATCHING = args.rotation_matching
OVERWRITE = args.overwrite

# PATHS
# Path("image-matching-challenge-2023")
data_dir = Path(args.data)
submission_dir = Path("submission.csv")
output_dir = Path(f"{args.output}/{CONF_NAME}")
output_dir.mkdir(exist_ok=True, parents=True)

metrics_path = output_dir / "metrics.pickle"
results_path = output_dir / "results.pickle"
submission_csv_path = output_dir / "submission.csv"

# CONFIG
config = configs[CONF_NAME]
with open(str(output_dir / "config.json"), "w") as jf:
    json.dump(config, jf, indent=4)

# if PIXSFM:
#     config["refinements"] = OmegaConf.load(pixsfm.configs.parse_config_path("low_memory"))

logging.info("CONFIG:")
for step, conf in config.items():
    if step == "n_retrieval":
        logging.info(f"  {step}: {conf}")
        continue

    if conf is None:
        logging.info(f"  {step}: None")
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
            logging.info("Skipping", dataset, scene)
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
            use_rotation_matching=ROTATION_MATCHING,
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


# WRITE SUBMISSION
with open(metrics_path, "rb") as handle:
    metrics = pickle.load(handle)
with open(results_path, "rb") as handle:
    out_results = pickle.load(handle)

create_submission(out_results, data_dict, submission_csv_path)
create_submission(out_results, data_dict, "submission.csv")

for dataset in metrics:
    logging.info(dataset)
    for scene in metrics[dataset]:
        logging.info(
            f"\t{scene}: {metrics[dataset][scene]['n_reg_images']} / {metrics[dataset][scene]['n_images']}"
        )

# EVALUATE
if MODE == "train":
    eval(submission_csv="submission.csv", data_dir=data_dir)
