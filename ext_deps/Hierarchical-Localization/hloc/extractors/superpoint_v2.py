import sys
from pathlib import Path

from ..utils.base_model import BaseModel

sys.path.append(str(Path(__file__).parent / "../../third_party"))
from sp_v2.sp_caps import SP_CAPS


class SuperpointV2(BaseModel):
    default_conf = {
        "backbone": {"blocks": [2, 2, 2]},
        "has_detector": True,
        "has_descriptor": True,
        # Inference
        "sparse_outputs": True,
        "nms_radius": 8,
        "refinement_radius": 3,
        "detection_threshold": 0.000,
        "max_num_keypoints": -1,
        "force_num_keypoints": False,
        "remove_borders": 4,
        # Descriptor
        "descriptor_dim": 128,
        "weights": None,
        # Homography adaptation
        "ha": {
            "enable": False,
            "num_H": 10,
            "mini_bs": 5,
            "aggregation": "mean",
            "H_params": {
                "difficulty": 0.8,
                "translation": 1.0,
                "max_angle": 60,
                "n_angles": 10,
                "min_convexity": 0.05,
            },
        },
    }
    required_data_keys = ["image"]

    required_inputs = ["image"]
    detection_noise = 2.0

    def _init(self, conf):
        self.net = SP_CAPS(conf)

    def _forward(self, data):
        out = self.net(data)

        # rename keypoint_scores to scores in data
        out["scores"] = out["keypoint_scores"]

        # swicht dims of descriptors
        out["descriptors"] = out["descriptors"].permute(0, 2, 1)

        return out
