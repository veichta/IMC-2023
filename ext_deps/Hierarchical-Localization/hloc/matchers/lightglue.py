import sys
from pathlib import Path

from ..utils.base_model import BaseModel

sys.path.append(str(Path(__file__).parent / "../../third_party/LightGlue"))
import torch
from model import LightGlue


# hloc interface
class LightGlueHloc(BaseModel):
    default_conf = {
        "weights": "superpoint_lightglue",
        "filter_threshold": 0.1,
        "flash": False,
        "input_dim": 256,
    }
    required_inputs = [
        "image0",
        "keypoints0",
        "scores0",
        "descriptors0",
        "image1",
        "keypoints1",
        "scores1",
        "descriptors1",
    ]

    def _init(self, conf):
        self.net = LightGlue(conf)  # torch.compile(LightGlue(conf))

    def _forward(self, data):
        return self.net(data)
