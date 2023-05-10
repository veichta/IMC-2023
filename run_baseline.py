import os
import sys
from pathlib import Path

import numpy as np
import pycolmap
from hloc import extract_features, match_features, match_dense, pairs_from_retrieval, reconstruction
import pickle
import json

sys.path.append("/kaggle/working/superglue/superglue")

# DIR = "/kaggle/input/image-matching-challenge-2023"
# DIR = "/cluster/project/infk/courses/252-0579-00L/group01/image-matching-challenge-2023"
DIR = "image-matching-challenge-2023"
MODE = "train"  # "train", "test"

# Configs

configs = {
    'sift+NN': {
        'features': extract_features.confs["sift"],
        'matches': match_features.confs["NN-ratio"],
        'retrieval': extract_features.confs["netvlad"],
        'n_retrieval': 50,

    },
    'loftr': {
        'features': None,
        'matches': {
            "output": "matches-loftr",
            "model": {"name": "loftr", "weights": "outdoor"},
            "preprocessing": {"grayscale": True, "resize_max": 840, "dfactor": 8},  # 1024,
            "max_error": 1,  # max error for assigned keypoints (in px)
            "cell_size": 1,  # size of quantization patch (max 1 kp/patch)
        },
        'retrieval': extract_features.confs["netvlad"],
        'n_retrieval': 20,
    }
}

NAME = "loftr"


# Data Dict
def get_data_from_dict():
    data_dict = {}
    with open(f"{DIR}/sample_submission.csv", "r") as f:
        for i, l in enumerate(f):
            # Skip header.
            if l and i > 0:
                image, dataset, scene, _, _ = l.strip().split(",")
                if dataset not in data_dict:
                    data_dict[dataset] = {}
                if scene not in data_dict[dataset]:
                    data_dict[dataset][scene] = []
                data_dict[dataset][scene].append(image)
    return data_dict


def get_data_from_dir():
    data_dict = {}

    datasets = [x for x in os.listdir(f"{DIR}/{MODE}") if os.path.isdir(f"{DIR}/{MODE}/{x}")]
    for dataset in datasets:
        if dataset not in data_dict:
            data_dict[dataset] = {}

        dataset_dir = f"{DIR}/{MODE}/{dataset}"
        scenes = [x for x in os.listdir(dataset_dir) if os.path.isdir(f"{dataset_dir}/{x}")]
        for scene in scenes:
            image_dir = f"{dataset_dir}/{scene}/images"
            data_dict[dataset][scene] = []
            for img in os.listdir(image_dir):
                data_dict[dataset][scene].append(os.path.join(dataset, scene, "images", img))

    return data_dict


data_dict = get_data_from_dict() if MODE == "test" else get_data_from_dir()

for dataset, ds_vals in data_dict.items():
    print(f"ds: {dataset}")
    for scene, imgs in ds_vals.items():
        print(f"  sc: {scene}")
        for img in imgs[:5]:
            print(f"     img: {img}")

dataset_to_scenes = {}

for ds, ds_vals in data_dict.items():
    dataset_to_scenes[ds] = ds_vals.keys()
    print(ds)
    for scene in dataset_to_scenes[ds]:
        print(f"  {scene}: {len(data_dict[ds][scene])} imgs")


# Submission Function
def arr_to_str(a):
    return ";".join([str(x) for x in a.reshape(-1)])


# Function to create a submission file.
def create_submission(out_results, data_dict, fname):
    with open(fname, "w") as f:
        f.write("image_path,dataset,scene,rotation_matrix,translation_vector\n")
        for dataset in data_dict:
            res = out_results[dataset] if dataset in out_results else {}
            for scene in data_dict[dataset]:
                scene_res = res[scene] if scene in res else {"R": {}, "t": {}}
                for image in data_dict[dataset][scene]:
                    if image in scene_res:
                        print(image)
                        R = np.array(scene_res[image]["R"]).reshape(-1)
                        T = np.array(scene_res[image]["t"]).reshape(-1)
                    else:
                        R = np.eye(3).reshape(-1)
                        T = np.zeros((3))
                    f.write(f"{image},{dataset},{scene},{arr_to_str(R)},{arr_to_str(T)}\n")
    f.close()


from dataclasses import dataclass
from time import time

# Eval train submission
# https://www.kaggle.com/code/eduardtrulls/imc2023-evaluation
import numpy as np


# Evaluation metric.
@dataclass
class Camera:
    rotmat: np.array
    tvec: np.array


def quaternion_from_matrix(matrix):
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    m00 = M[0, 0]
    m01 = M[0, 1]
    m02 = M[0, 2]
    m10 = M[1, 0]
    m11 = M[1, 1]
    m12 = M[1, 2]
    m20 = M[2, 0]
    m21 = M[2, 1]
    m22 = M[2, 2]

    # Symmetric matrix K.
    K = np.array(
        [
            [m00 - m11 - m22, 0.0, 0.0, 0.0],
            [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
            [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
            [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
        ]
    )
    K /= 3.0

    # Quaternion is eigenvector of K that corresponds to largest eigenvalue.
    w, V = np.linalg.eigh(K)
    q = V[[3, 0, 1, 2], np.argmax(w)]

    if q[0] < 0.0:
        np.negative(q, q)
    return q


def evaluate_R_t(R_gt, t_gt, R, t, eps=1e-15):
    t = t.flatten()
    t_gt = t_gt.flatten()

    q_gt = quaternion_from_matrix(R_gt)
    q = quaternion_from_matrix(R)
    q = q / (np.linalg.norm(q) + eps)
    q_gt = q_gt / (np.linalg.norm(q_gt) + eps)
    loss_q = np.maximum(eps, (1.0 - np.sum(q * q_gt) ** 2))
    err_q = np.arccos(1 - 2 * loss_q)

    GT_SCALE = np.linalg.norm(t_gt)
    t = GT_SCALE * (t / (np.linalg.norm(t) + eps))
    err_t = min(np.linalg.norm(t_gt - t), np.linalg.norm(t_gt + t))

    return np.degrees(err_q), err_t


def compute_dR_dT(R1, T1, R2, T2):
    """Given absolute (R, T) pairs for two cameras, compute the relative pose difference, from the first."""

    dR = np.dot(R2, R1.T)
    dT = T2 - np.dot(dR, T1)
    return dR, dT


def compute_mAA(err_q, err_t, ths_q, ths_t):
    """Compute the mean average accuracy over a set of thresholds. Additionally returns the metric only over rotation and translation."""

    acc, acc_q, acc_t = [], [], []
    for th_q, th_t in zip(ths_q, ths_t):
        cur_acc_q = err_q <= th_q
        cur_acc_t = err_t <= th_t
        cur_acc = cur_acc_q & cur_acc_t

        acc.append(cur_acc.astype(np.float32).mean())
        acc_q.append(cur_acc_q.astype(np.float32).mean())
        acc_t.append(cur_acc_t.astype(np.float32).mean())
    return np.array(acc), np.array(acc_q), np.array(acc_t)


def dict_from_csv(csv_path, has_header):
    csv_dict = {}
    with open(csv_path, "r") as f:
        for i, l in enumerate(f):
            if has_header and i == 0:
                continue
            if l:
                image, dataset, scene, R_str, T_str = l.strip().split(",")
                R = np.fromstring(R_str.strip(), sep=";").reshape(3, 3)
                T = np.fromstring(T_str.strip(), sep=";")
                if dataset not in csv_dict:
                    csv_dict[dataset] = {}
                if scene not in csv_dict[dataset]:
                    csv_dict[dataset][scene] = {}
                csv_dict[dataset][scene][image] = Camera(rotmat=R, tvec=T)
    return csv_dict


def eval_submission(
    submission_csv_path,
    ground_truth_csv_path,
    rotation_thresholds_degrees_dict,
    translation_thresholds_meters_dict,
    verbose=False,
):
    """Compute final metric given submission and ground truth files. Thresholds are specified per dataset."""

    submission_dict = dict_from_csv(submission_csv_path, has_header=True)
    gt_dict = dict_from_csv(ground_truth_csv_path, has_header=True)

    # Check that all necessary keys exist in the submission file
    for dataset in gt_dict:
        assert dataset in submission_dict, f"Unknown dataset: {dataset}"
        for scene in gt_dict[dataset]:
            assert scene in submission_dict[dataset], f"Unknown scene: {dataset}->{scene}"
            for image in gt_dict[dataset][scene]:
                assert (
                    image in submission_dict[dataset][scene]
                ), f"Unknown image: {dataset}->{scene}->{image}"

    # Iterate over all the scenes
    if verbose:
        t = time()
        print("*** METRICS ***")

    metrics_per_dataset = []
    for dataset in gt_dict:
        metrics_per_scene = []
        for scene in gt_dict[dataset]:
            err_q_all = []
            err_t_all = []
            images = list(gt_dict[dataset][scene])
            # Process all pairs in a scene
            for i in range(len(images)):
                for j in range(i + 1, len(images)):
                    gt_i = gt_dict[dataset][scene][images[i]]
                    gt_j = gt_dict[dataset][scene][images[j]]
                    dR_gt, dT_gt = compute_dR_dT(gt_i.rotmat, gt_i.tvec, gt_j.rotmat, gt_j.tvec)

                    pred_i = submission_dict[dataset][scene][images[i]]
                    pred_j = submission_dict[dataset][scene][images[j]]
                    dR_pred, dT_pred = compute_dR_dT(
                        pred_i.rotmat, pred_i.tvec, pred_j.rotmat, pred_j.tvec
                    )

                    err_q, err_t = evaluate_R_t(dR_gt, dT_gt, dR_pred, dT_pred)
                    err_q_all.append(err_q)
                    err_t_all.append(err_t)

            mAA, mAA_q, mAA_t = compute_mAA(
                err_q=err_q_all,
                err_t=err_t_all,
                ths_q=rotation_thresholds_degrees_dict[(dataset, scene)],
                ths_t=translation_thresholds_meters_dict[(dataset, scene)],
            )
            if verbose:
                print(
                    f"{dataset} / {scene} ({len(images)} images, {len(err_q_all)} pairs) -> mAA={np.mean(mAA):.06f}, mAA_q={np.mean(mAA_q):.06f}, mAA_t={np.mean(mAA_t):.06f}"
                )
            metrics_per_scene.append(np.mean(mAA))

        metrics_per_dataset.append(np.mean(metrics_per_scene))
        if verbose:
            print(f"{dataset} -> mAA={np.mean(metrics_per_scene):.06f}")
            print()

    if verbose:
        print(f"Final metric -> mAA={np.mean(metrics_per_dataset):.06f} (t: {time() - t} sec.)")
        print()

    return np.mean(metrics_per_dataset)

# create_submission(out_results, data_dict)
ds = [str(d) for d in data_dict.keys()] if MODE == "test" else ["heritage", "haiper", "urban"]

conf = configs[NAME]

out_dir = Path(f'outputs/{NAME}')
out_dir.mkdir(exist_ok=True, parents=True)

metrics_path = out_dir / 'metrics.pickle'
results_path = out_dir / 'results.pickle'

submission_csv_path = out_dir / "submission.csv"

with open(str(out_dir / "config.json"), "w") as jf:
    json.dump(conf, jf, indent=4)

# Setup
metrics = {}

out_results = {}

# Run
for dataset in ds:
    print(dataset)

    if dataset not in metrics:
        metrics[dataset] = {}

    if dataset not in out_results:
        out_results[dataset] = {}

    scenes = dataset_to_scenes[dataset]
    for scene in data_dict[dataset]:
        print(f"{dataset} - {scene}")

        image_dir = Path(f"{DIR}/{MODE}/{dataset}/{scene}/images")
        if not os.path.exists(image_dir):
            print("Skipping", dataset, scene)
            continue

        scene_dir = out_dir / dataset / scene
        scene_dir.mkdir(parents=True, exist_ok=True)

        sfm_dir = scene_dir / "sparse"
        pairs_path = scene_dir / 'pairs.txt'
        features_retrieval = scene_dir / "features_retrieval.h5"
        features_path = scene_dir / "features.h5"
        matches_path = scene_dir / 'matches.h5'

        img_list = [Path(p).name for p in data_dict[dataset][scene]]

        out_results[dataset][scene] = {}

        # exhaustive retrieval
        extract_features.main(
            conf=conf['retrieval'],
            image_dir=image_dir,
            image_list=img_list,
            feature_path=features_retrieval,
        )

        pairs_from_retrieval.main(
            descriptors=features_retrieval,
            num_matched=min(len(img_list), conf['n_retrieval']),
            output=pairs_path,
        )

        if conf['features'] is not None:
            # feature extraction
            extract_features.main(
                conf=conf['features'],
                image_dir=image_dir,
                image_list=img_list,
                feature_path=features_path,
            )

            # feature matching
            match_features.main(
                conf=conf['matches'],
                pairs=pairs_path,
                features=features_path,
                matches=matches_path,
            )
        else:
            match_dense.main(
            conf=conf['matches'],
            image_dir=image_dir,
            pairs=pairs_path,
            features=features_path,
            matches=matches_path,
        )

        # structure-from-motion
        if sfm_dir.exists():
            try:
                sparse_model = pycolmap.Reconstruction(sfm_dir)
            except ValueError:
                sparse_model = None
        else:
            sparse_model = reconstruction.main(
                sfm_dir=sfm_dir,
                image_dir=image_dir,
                image_list=img_list,
                pairs=pairs_path,
                features=features_path,
                matches=matches_path,
                verbose=False,
            )
            if sparse_model is not None:
                sparse_model.write(sfm_dir)

        if sparse_model is None:
            print(f"No model reconstructed for {dataset} - {scene}.")
            metrics[dataset][scene] = {
                "n_images": len(img_list),
                "n_reg_images": 0,
            }
            continue

        metrics[dataset][scene] = {
            "n_images": len(img_list),
            "n_reg_images": sparse_model.num_reg_images(),
        }

        # save results of current scene
        print(f"Extracting results for {dataset} - {scene}")
        for _, im in sparse_model.images.items():
            # print(im)
            img_name = os.path.join(dataset, scene, "images", im.name)
            # problem: tvec is a reference! --> force copy
            out_results[dataset][scene][img_name] = {"R": im.rotmat(), "t": np.array(im.tvec)}
        print("Done...")

        with open(metrics_path, 'wb') as handle:
            pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(results_path, 'wb') as handle:
            pickle.dump(out_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(metrics_path, 'rb') as handle:
    metrics = pickle.load(handle)
with open(results_path, 'rb') as handle:
    out_results = pickle.load(handle)

for dataset in metrics:
    print(dataset)
    for scene in metrics[dataset]:
        print(
            f"\t{scene}: {metrics[dataset][scene]['n_reg_images']} / {metrics[dataset][scene]['n_images']}"
        )

create_submission(out_results, data_dict, submission_csv_path)

# Set rotation thresholds per scene.

rotation_thresholds_degrees_dict = {
    **{("haiper", scene): np.linspace(1, 10, 10) for scene in ["bike", "chairs", "fountain"]},
    **{("heritage", scene): np.linspace(1, 10, 10) for scene in ["cyprus", "dioscuri"]},
    **{("heritage", "wall"): np.linspace(0.2, 10, 10)},
    **{("urban", "kyiv-puppet-theater"): np.linspace(1, 10, 10)},
}

translation_thresholds_meters_dict = {
    **{("haiper", scene): np.geomspace(0.05, 0.5, 10) for scene in ["bike", "chairs", "fountain"]},
    **{("heritage", scene): np.geomspace(0.1, 2, 10) for scene in ["cyprus", "dioscuri"]},
    **{("heritage", "wall"): np.geomspace(0.05, 1, 10)},
    **{("urban", "kyiv-puppet-theater"): np.geomspace(0.5, 5, 10)},
}

# Generate and evaluate a random submission.

if MODE == "train":
    with open(f"{DIR}/train/train_labels.csv", "r") as fr, open("ground_truth.csv", "w") as fw:
        for i, l in enumerate(fr):
            if i == 0:
                fw.write("image_path,dataset,scene,rotation_matrix,translation_vector\n")
            else:
                dataset, scene, image, R, T = l.strip().split(",")
                fw.write(f"{image},{dataset},{scene},{R},{T}\n")

    eval_submission(
        submission_csv_path=submission_csv_path,
        ground_truth_csv_path="ground_truth.csv",
        rotation_thresholds_degrees_dict=rotation_thresholds_degrees_dict,
        translation_thresholds_meters_dict=translation_thresholds_meters_dict,
        verbose=True,
    )
