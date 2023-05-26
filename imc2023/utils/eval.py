import argparse
import logging
from dataclasses import dataclass
from time import time

import numpy as np

# Eval train submission
# https://www.kaggle.com/code/eduardtrulls/imc2023-evaluation


# Conveniency functions.
def arr_to_str(a):
    return ";".join([str(x) for x in a.reshape(-1)])


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
    return_dict=False,
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
        logging.info("*** METRICS ***")

    all_metrics = {}
    metrics_per_dataset = []
    for dataset in gt_dict:
        metrics_per_scene = []
        all_metrics[dataset] = {}
        for scene in gt_dict[dataset]:
            all_metrics[dataset][scene] = {}
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
                logging.info(
                    f"{dataset} / {scene} ({len(images)} images, {len(err_q_all)} pairs) -> "
                    + f"mAA={np.mean(mAA):.06f}, "
                    + f"mAA_q={np.mean(mAA_q):.06f}, "
                    + f"mAA_t={np.mean(mAA_t):.06f}"
                )
            metrics_per_scene.append(np.mean(mAA))
            all_metrics[dataset][scene]["mAA"] = float(np.mean(mAA))
            all_metrics[dataset][scene]["mAA_q"] = float(np.mean(mAA_q))
            all_metrics[dataset][scene]["mAA_t"] = float(np.mean(mAA_t))

        metrics_per_dataset.append(np.mean(metrics_per_scene))
        all_metrics[dataset]["mAA"] = float(np.mean(metrics_per_scene))

        if verbose:
            logging.info(f"{dataset} -> mAA={np.mean(metrics_per_scene):.06f}")

    if verbose:
        logging.info(
            f"Final metric -> mAA={np.mean(metrics_per_dataset):.06f} (t: {time() - t} sec.)"
        )

    all_metrics["mAA"] = float(np.mean(metrics_per_dataset))
    if return_dict:
        return all_metrics

    return np.mean(metrics_per_dataset)


def eval(
    submission_csv: str, data_dir: str, verbose: bool = True, return_dict: bool = False
) -> float:
    """Evaluate submission.

    Args:
        submission_csv (str): Path to submission csv file.
        data_dir (str): Path to data directory.
        verbose (bool): Whether to logging.info metrics. Defaults to True.
        return_dict (bool): Whether to return a dictionary with all metrics. Defaults to False.

    Returns:
        float: Mean average accuracy.
    """
    # Set rotation thresholds per scene.
    rotation_thresholds_degrees_dict = {
        **{("haiper", scene): np.linspace(1, 10, 10) for scene in ["bike", "chairs", "fountain"]},
        **{("heritage", scene): np.linspace(1, 10, 10) for scene in ["cyprus", "dioscuri"]},
        **{("heritage", "wall"): np.linspace(0.2, 10, 10)},
        **{("urban", "kyiv-puppet-theater"): np.linspace(1, 10, 10)},
    }

    translation_thresholds_meters_dict = {
        **{
            ("haiper", scene): np.geomspace(0.05, 0.5, 10)
            for scene in ["bike", "chairs", "fountain"]
        },
        **{("heritage", scene): np.geomspace(0.1, 2, 10) for scene in ["cyprus", "dioscuri"]},
        **{("heritage", "wall"): np.geomspace(0.05, 1, 10)},
        **{("urban", "kyiv-puppet-theater"): np.geomspace(0.5, 5, 10)},
    }

    # Generate GT.
    with open(f"{data_dir}/train/train_labels.csv", "r") as fr, open("ground_truth.csv", "w") as fw:
        for i, l in enumerate(fr):
            if i == 0:
                fw.write("image_path,dataset,scene,rotation_matrix,translation_vector\n")
            else:
                dataset, scene, image, R, T = l.strip().split(",")
                fw.write(f"{image},{dataset},{scene},{R},{T}\n")

    return eval_submission(
        submission_csv_path=submission_csv,
        ground_truth_csv_path="ground_truth.csv",
        rotation_thresholds_degrees_dict=rotation_thresholds_degrees_dict,
        translation_thresholds_meters_dict=translation_thresholds_meters_dict,
        verbose=verbose,
        return_dict=return_dict,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=str, default="submission.csv")
    parser.add_argument("--dir", type=str, default=".")
    args = parser.parse_args()

    eval(args.submission, args.dir)
