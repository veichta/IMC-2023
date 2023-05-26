import datetime
import logging
import os

import cv2
import numpy as np
from dioad.utils import rotate_preserve_size

logger = logging.getLogger("transformers")
logger.setLevel(logging.ERROR)

from transformers import ViTFeatureExtractor

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")


def preprocess(model_name, image_path):
    if model_name in ["vit", "tag-cnn"]:
        image_size = 224
    else:
        image_size = 299

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size))
    img = np.array(img)

    if model_name == "vit":
        X_vit = [img]
        X_vit = feature_extractor(images=X_vit, return_tensors="pt")["pixel_values"]
        X_vit = np.array(X_vit)
        X = X_vit

    if model_name == "tag-cnn":
        X_vit = [img]
        X_vit = feature_extractor(images=X_vit, return_tensors="pt")["pixel_values"]
        X_vit = np.array(X_vit)

        img = np.expand_dims(img, axis=0)
        X = [X_vit, img]

    if model_name in ["efficientnetv2b2", "en", "efficientnetv2b2"]:
        img = np.expand_dims(img, axis=0)
        X = img

    return X


def postprocess(img_path, angle, image_size, save_image_dir):
    img = rotate_preserve_size(img_path, angle, (image_size, image_size), False)

    # filename = "cs776a-pred.jpg" #img_path.split("/")[-1]
    filename = "pred_" + img_path.split("/")[-1]

    try:
        img.save(os.path.join(save_image_dir, filename))
    except:
        filename = str(datetime.datetime.now()) + "_" + filename
        img.save(os.path.join(save_image_dir, filename))
