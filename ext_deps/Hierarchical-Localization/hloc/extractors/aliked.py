import sys
from pathlib import Path
import torch
from types import SimpleNamespace

from ..utils.base_model import BaseModel

aliked_path = Path(__file__).parent / "../../third_party/ALIKED"
sys.path.append(str(aliked_path))

from nets.aliked import ALIKED as ALIKED_

class ALIKED(BaseModel):
    default_conf = {
        'model_name': 'aliked-n16',  # 'aliked-t16', 'aliked-n16', 'aliked-n16rot', 'aliked-n32'
        'max_num_keypoints': -1,
        'detection_threshold': 0.0,
        'force_num_keypoints': False,
    }
    required_data_keys = ['image']

    def _init(self, conf):
        print(conf)
        conf = self.conf = SimpleNamespace(**conf)
        self.model = ALIKED_(
            model_name=conf.model_name,
            device='cuda',
            top_k=conf.max_num_keypoints,
            scores_th=conf.detection_threshold,
            n_limit=5000)

    def _forward(self, data):
        pred = self.model(data['image'])
        pred = {k: torch.stack(v) for k, v in pred.items() if isinstance(v, list)}

        _, _, h, w = data['image'].shape
        wh = torch.tensor([w - 1, h - 1], device=data['image'].device)
        pred['keypoints'] = wh*(pred['keypoints'] + 1) / 2.0
        pred['descriptors'] = pred['descriptors'].transpose(-1, -2)
        pred['scores'] = pred.pop('scores')
        return pred

    def loss(self, pred, data):
        raise NotImplementedError

    def metrics(self, pred, data):
        raise NotImplementedError