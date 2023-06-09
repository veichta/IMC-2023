import sys

# for Euler
sys.path.append("ext_deps/Hierarchical-Localization")

# for Kaggle
sys.path.append("/kaggle/input/imc-23-repo/IMC-2023/ext_deps/Hierarchical-Localization")

import argparse
import logging
from pathlib import Path

import pycolmap
from hloc import reconstruction

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, help="Path to the model directory.")
parser.add_argument("--image_dir", type=str, help="Path to the directory with images.")
parser.add_argument("--pairs_path", type=str, help="Path to the pairs file.")
parser.add_argument("--keypoints_path", type=str, help="Path to the keypoints file.")
parser.add_argument("--matches_path", type=str, help="Path to the matches file.")
parser.add_argument("--camera_mode", type=str, choices=["single", "auto"], help="Camera mode.")
args = parser.parse_args()

formatter = logging.Formatter(
    fmt="[%(asctime)s %(name)s %(levelname)s] %(message)s", datefmt="%Y/%m/%d %H:%M:%S"
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
handler.setLevel(logging.DEBUG)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)
logger.propagate = False


logging.info(f"Running reconstruction with python: {sys.executable}")

camera_mode = (
    pycolmap.CameraMode.SINGLE if args.camera_mode == "single" else pycolmap.CameraMode.AUTO
)

mapper_options = pycolmap.IncrementalMapperOptions()
mapper_options.min_model_size = 6

reconstruction.main(
    sfm_dir=Path(args.model_path),
    image_dir=Path(args.image_dir),
    pairs=Path(args.pairs_path),
    features=Path(args.keypoints_path),
    matches=Path(args.matches_path),
    mapper_options=mapper_options.todict(),
    camera_mode=camera_mode,
    verbose=False,
)
