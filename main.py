import sys

sys.path.append("ext_deps/Hierarchical-Localization")
sys.path.append("ext_deps/dioad")

import argparse
import datetime
import json
import logging
import os
import pickle
import shutil
import time
from pathlib import Path

import numpy as np
from hloc import extract_features

from imc2023.configs import configs
from imc2023.utils.eval import eval
from imc2023.utils.utils import (
    DataPaths,
    create_submission,
    get_data_from_dict,
    get_data_from_dir,
    setup_logger,
)

## Kaggle version
# import sys

# sys.path.append("/kaggle/input/imc-23-repo/IMC-2023")
# sys.path.append("/kaggle/input/imc-23-repo/IMC-2023/ext_deps/Hierarchical-Localization")
# sys.path.append("/kaggle/input/imc-23-repo/IMC-2023/ext_deps/dioad")

# import os
# import argparse

# from main import main

# args = {
#     "data": "/kaggle/input/image-matching-challenge-2023",
#     "configs": ["SP+LG"],
#     "retrieval": "netvlad",
#     "n_retrieval": 50,
#     "mode": "train",
#     "output": "/kaggle/temp",
#     "pixsfm": True,
#     "pixsfm_max_imgs": 9999,
#     "pixsfm_config": "low_memory",
#     "pixsfm_script_path": "/kaggle/input/imc-23-repo/IMC-2023/run_pixsfm.py",
#     "rotation_matching": True,
#     "rotation_wrapper": False,
#     "cropping": False,
#     "max_rel_crop_size": 0.75,
#     "min_rel_crop_size": 0.2,
#     "resize": None,
#     "shared_camera": True,
#     "overwrite": True,
#     "kaggle": True,
#     "skip_scenes": None,
# }

# args = argparse.Namespace(**args)
# os.makedirs(args.output, exist_ok=True)

# main(args)


def get_output_dir(args: argparse.Namespace) -> Path:
    """Get the output directory.

    Args:
        args (argparse.Namespace): Arguments.

    Returns:
        Path: Output directory.
    """
    name = "+".join(sorted(list(args.configs)))
    output_dir = f"{args.output}/{name}"
    if args.rotation_matching:
        output_dir += "-rot"
    if args.rotation_wrapper:
        output_dir += "-rotwrap"
    if args.pixsfm:
        output_dir += "-pixsfm"
    if args.cropping:
        output_dir += "-crop"
    if args.resize is not None:
        output_dir += f"-{args.resize}px"
    if args.shared_camera:
        output_dir += "-sci"

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    return output_dir


def log_args(args: argparse.Namespace) -> None:
    """Log arguments.

    Args:
        args (argparse.Namespace): Arguments.
    """
    logging.info("=" * 80)
    logging.info("Arguments:")
    logging.info("=" * 80)
    for arg in vars(args):
        logging.info(f"  {arg}: {getattr(args, arg)}")


def main(args):
    start = time.time()
    setup_logger()
    log_args(args)

    # PATHS
    data_dir = Path(args.data)
    output_dir = get_output_dir(args)

    metrics_path = output_dir / "metrics.pickle"
    results_path = output_dir / "results.pickle"
    submission_csv_path = Path(f"{output_dir}/submission.csv")

    # CONFIG
    assert len(args.configs) > 0, "No configs specified"
    confs = [configs[c] for c in args.configs]

    config = {
        "features": [c["features"] for c in confs],
        "matches": [c["matches"] for c in confs],
        "retrieval": extract_features.confs[args.retrieval],
        "n_retrieval": args.n_retrieval,
    }
    with open(str(output_dir / "config.json"), "w") as jf:
        json.dump(config, jf, indent=4)

    # SETUP DATA DICT
    data_dict = (
        get_data_from_dict(data_dir)
        if args.mode == "test"
        else get_data_from_dir(data_dir, args.mode)
    )

    # RUN
    metrics = {}
    out_results = {}

    if config["features"] is not None:
        from imc2023.pipelines.sparse_pipeline import SparsePipeline as Pipeline
    else:
        from imc2023.pipelines.dense_pipeline import DensePipeline as Pipeline

    for dataset in data_dict:
        logging.info("=" * 80)
        logging.info(dataset)
        logging.info("=" * 80)

        if dataset not in metrics:
            metrics[dataset] = {}

        if dataset not in out_results:
            out_results[dataset] = {}

        for scene in data_dict[dataset]:
            logging.info("=" * 80)
            logging.info(f"{dataset} - {scene}")
            logging.info("=" * 80)

            if args.skip_scenes is not None and scene in args.skip_scenes and args.mode == "train":
                logging.info(f"Skipping {dataset} - {scene}")
                continue

            start_scene = time.time()

            # SETUP PATHS
            paths = DataPaths(
                data_dir=data_dir,
                output_dir=output_dir,
                dataset=dataset,
                scene=scene,
                mode=args.mode,
            )

            img_list = [Path(p).name for p in data_dict[dataset][scene]]
            out_results[dataset][scene] = {}

            if not paths.image_dir.exists():
                logging.info(f"Skipping {dataset} - {scene} (no images)")
                continue

            # Define and run pipeline
            pipeline = Pipeline(config=config, paths=paths, img_list=img_list, args=args)

            pipeline.run()

            sparse_model = pipeline.sparse_model

            if sparse_model is None:
                continue

            logging.info(f"Extracting results for {dataset} - {scene}")
            for _, im in sparse_model.images.items():
                img_name = os.path.join(dataset, scene, "images", im.name)
                # problem: tvec is a reference! --> force copy
                out_results[dataset][scene][img_name] = {"R": im.rotmat(), "t": np.array(im.tvec)}

            metrics[dataset][scene] = {
                "n_images": len(img_list),
                "n_reg_images": sparse_model.num_reg_images(),
            }

            total_time = time.time() - start_scene
            logging.info("Timings:")
            for k, v in pipeline.timing.items():
                logging.info(f"  {k}: {datetime.timedelta(seconds=v)} ({v / total_time:.2%})")
            logging.info(f"  Total: {datetime.timedelta(seconds=total_time)}")

            timings_path = paths.scene_dir / "timings.json"
            with open(timings_path, "w") as jf:
                json.dump(pipeline.timing, jf, indent=4)

            with open(metrics_path, "wb") as handle:
                pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(results_path, "wb") as handle:
                pickle.dump(out_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # delete scene dir
            if args.mode == "test":
                shutil.rmtree(paths.scene_dir)

    create_submission(out_results, data_dict, submission_csv_path)
    create_submission(out_results, data_dict, "submission.csv")

    if args.mode == "train":
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

    logging.info(f"Done in {datetime.timedelta(seconds=time.time() - start)}")


if __name__ == "__main__":
    # PARSE ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="imc dataset")
    parser.add_argument(
        "--configs", type=str, required=True, nargs="+", choices=configs.keys(), help="configs"
    )
    parser.add_argument("--retrieval", type=str, required=True, choices=["netvlad", "cosplace"])
    parser.add_argument("--n_retrieval", type=int, default=50, help="number of retrieval images")
    parser.add_argument(
        "--mode", type=str, required=True, choices=["train", "test"], help="train or test"
    )
    parser.add_argument("--output", type=str, default="outputs", help="output dir")
    parser.add_argument("--pixsfm", action="store_true", help="use pixsfm")
    parser.add_argument(
        "--pixsfm_max_imgs",
        type=int,
        default=9999,
        help="max number of images for PixSfM",
    )
    parser.add_argument(
        "--pixsfm_low_mem_threshold",
        type=int,
        default=50,
        required=True,
        help="low mem threshold for PixSfM",
    )
    parser.add_argument("--pixsfm_config", type=str, default="low_memory", help="PixSfM config")
    parser.add_argument(
        "--pixsfm_script_path", type=str, default="run_pixsfm.py", help="PixSfM script path"
    )
    parser.add_argument("--rotation_matching", action="store_true", help="use rotation matching")
    parser.add_argument(
        "--rotation_wrapper",
        action="store_true",
        help="wrapper implementation of rotation matching",
    )
    parser.add_argument("--cropping", action="store_true", help="use image cropping")
    parser.add_argument(
        "--max_rel_crop_size", type=float, default=0.75, help="EITHER crop must have a smaller relative size"
    )
    parser.add_argument(
        "--min_rel_crop_size", type=float, default=0.2, help="BOTH crops must have a larger relative size"
    )
    parser.add_argument("--resize", type=int, help="resize images")
    parser.add_argument("--shared_camera", action="store_true", help="use shared camera intrinsics")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing results")
    parser.add_argument("--kaggle", action="store_true", help="kaggle mode")
    parser.add_argument("--skip_scenes", nargs="+", help="scenes to skip")
    args = parser.parse_args()

    main(args)
