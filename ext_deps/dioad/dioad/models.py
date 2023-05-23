import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras.models import Model
from transformers import TFAutoModel


def create_vit_architecture():
    IMAGE_SIZE = 224
    vit_base = TFAutoModel.from_pretrained("google/vit-base-patch16-224")

    img_input = L.Input(shape=(3, IMAGE_SIZE, IMAGE_SIZE))
    x = vit_base(img_input)
    y = L.Dense(1, activation="linear")(x[-1])

    model = Model(img_input, y)
    # print(model.summary())
    return model


def load_vit_model(load_pretrained_weights=True, load_model_path="weights/model-vit-ang-loss.h5"):
    model = create_vit_architecture()
    if load_pretrained_weights:
        model.load_weights(load_model_path)
    return model
