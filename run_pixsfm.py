import sys

# for Euler
sys.path.append("ext_deps/Hierarchical-Localization")

# for Kaggle
sys.path.append("/kaggle/input/imc-23-repo/IMC-2023/ext_deps/Hierarchical-Localization")

# TODO: Remove this line
sys.path.append("/kaggle/input/imc-23-repo-tmp/IMC-2023-TMP/ext_deps/Hierarchical-Localization")

import argparse
from pathlib import Path

import pixsfm
import pycolmap
from omegaconf import OmegaConf
from pixsfm.refine_hloc import PixSfM

parser = argparse.ArgumentParser()
parser.add_argument("--sfm_dir", type=str)
parser.add_argument("--image_dir", type=str)
parser.add_argument("--pairs_path", type=str)
parser.add_argument("--features_path", type=str)
parser.add_argument("--matches_path", type=str)
parser.add_argument("--cache_path", type=str)
parser.add_argument("--pixsfm_config", type=str)
parser.add_argument("--camera_mode", type=str, choices=["single", "auto"])
args = parser.parse_args()

if args.camera_mode == "single":
    camera_mode = pycolmap.CameraMode.SINGLE
else:
    camera_mode = pycolmap.CameraMode.AUTO

refiner = PixSfM(conf=OmegaConf.load(pixsfm.configs.parse_config_path(args.pixsfm_config)))
sparse_model, _ = refiner.run(
    output_dir=Path(args.sfm_dir),
    image_dir=Path(args.image_dir),
    pairs_path=Path(args.pairs_path),
    features_path=Path(args.features_path),
    matches_path=Path(args.matches_path),
    cache_path=Path(args.cache_path),
    verbose=False,
    camera_mode=camera_mode,
)

if sparse_model is not None:
    sparse_model.write(Path(args.sfm_dir))

# clear cache
for file in Path(args.cache_path).glob("*"):
    file.unlink()
