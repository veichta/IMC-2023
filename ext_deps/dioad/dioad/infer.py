import argparse

from dioad.models import load_vit_model
from dioad.processing import postprocess, preprocess


class Inference:
    def __init__(self, load_model_path="weights/model-vit-ang-loss.h5"):
        self.vit_model = load_vit_model(load_model_path=load_model_path)

    def predict(self, model_name, image_path, save_image_dir=None):
        X = preprocess(model_name, image_path)

        y = self.vit_model.predict(X, verbose=0)[0][0]

        pred_angle = -y
        if save_image_dir:
            postprocess(image_path, pred_angle, 400, save_image_dir)
        return y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="vit")
    parser.add_argument("--image-path", type=str, required=True)
    parser.add_argument(
        "--model-path", type=str, required=False, default="weights/model-vit-ang-loss.h5"
    )
    args = parser.parse_args()

    model = Inference(load_model_path=args.model_path)
    model.predict(args.model_name, args.image_path)
