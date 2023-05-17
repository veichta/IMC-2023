## Minimal wheels for python inference

# Usage
```shell
python dioad/infer.py --image-path=inputs/image.jpg --model-path=C:\Users\andro\Downloads\model-vit-ang-loss.h5
```

```python
import dioad.infer
deep_orientation = dioad.infer.Inference(load_model_path="/kaggle/input/orientation-prediction-model/model-vit-ang-loss.h5")

import glob
import cv2
import matplotlib.pyplot as plt
from dioad.processing import rotate_preserve_size

path_list = glob.glob("/kaggle/input/image-matching-challenge-2023/train/heritage/cyprus/images/*.JPG")
for path in path_list[:10]:
    angle = deep_orientation.predict("vit", path)
    img = cv2.imread(path)
    img = rotate_preserve_size(path, -angle, (400, 400), False)
    plt.imshow(img)
    plt.show()
```
