import argparse
import gc
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
from hloc import extract_features, match_features, pairs_from_exhaustive
from hloc.utils.io import list_h5_names, get_matches, get_keypoints

import os
import numpy as np
from tqdm import tqdm
import cv2
import networkx as nx
import dioad.infer

def get_match_matrix(features, matches):
    image_names = sorted(list_h5_names(features))
    pairs = sorted(list_h5_names(matches))
    match_matrix = np.zeros([len(image_names), len(image_names)])

    for pair in tqdm(pairs):
        name0, name1 = pair.split("/")
        m, sc = get_matches(matches, name0, name1)
        i = image_names.index(name0)
        j = image_names.index(name1)
        match_matrix[j, i] = match_matrix[i, j] = m.shape[0]
    return match_matrix, image_names

def estimate_rot(name0,name1, features, matches):
    kp0, kp1 = get_keypoints(features, name0), get_keypoints(features, name1)
    m, sc = get_matches(matches, name0, name1)
    src_pts = np.array(kp0[m[:,0]], dtype=np.float32)
    dst_pts = np.array(kp1[m[:,1]], dtype=np.float32)
    pts = np.stack((src_pts, dst_pts))
    M, inliers = cv2.estimateAffine2D(
            src_pts, dst_pts, method=cv2.RANSAC
        )
    return np.arctan2(M[1, 0], M[0, 0])

def estimate_rot_from_sift(features, matches):
    match_matrix, image_names = get_match_matrix(features, matches)
    n = match_matrix.shape[0]

    # compute maximum spanning tree
    # Create a weighted graph using networkx
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            weight = match_matrix[i, j]
            G.add_edge(i, j, weight=weight)
    print(G)
    # Compute the maximum spanning tree
    max_spanning_tree = nx.maximum_spanning_tree(G)

    tree = nx.DiGraph() 
    tree.add_nodes_from(range(n))
    for (i,j,r) in max_spanning_tree.edges(data=True):
        r = round(estimate_rot(image_names[i],image_names[j],features,matches)/(np.pi/4))*np.pi/4
        tree.add_edge(i,j, rotation =r)
        tree.add_edge(j,i, rotation=-r)
    


    reference_image = 0  # Choose a reference image
    rotation_values = np.zeros(n)  # Initialize rotation values
    visited = np.zeros(n)
    def propagate_rotation(current_image, accumulated_rotation):
        rotation_values[current_image] = accumulated_rotation
        neighbors = tree.neighbors(current_image)
        if visited[current_image]:
            return
        visited[current_image]=1
        for neighbor in neighbors:
            relative_rotation = tree.edges[current_image, neighbor]['rotation']
            propagate_rotation(neighbor, accumulated_rotation + relative_rotation)

    propagate_rotation(0,0)
    #rotation_values = rotation_values % np.pi*2
    q = (np.round(rotation_values*2/np.pi)%4)*90
    q-=np.median(q)
    rotation_angles = dict(zip(image_names,q))
    return rotation_angles

def resize_image(image: np.ndarray, max_size: int) -> np.ndarray:
    """Resize image to max_size.

    Args:
        image (np.ndarray): Image to resize.
        max_size (int): Maximum size of the image.

    Returns:
        np.ndarray: Resized image.
    """
    img_size = image.shape[:2]
    if max(img_size) > max_size:
        ratio = max_size / max(img_size)
        image = cv2.resize(image, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_LANCZOS4)
    return image


def get_rotated_image(image: np.ndarray, angle: float) -> Tuple[np.ndarray, float]:
    """Rotate image by angle (rounded to 90 degrees).

    Args:
        image (np.ndarray): Image to rotate.
        angle (float): Angle to rotate the image by.

    Returns:
        Tuple[np.ndarray, float]: Rotated image and the angle it was rotated by.
    """
    if angle < 0.0:
        angle += 360
    angle = (round(angle / 90.0) * 90) % 360  # angle is now an integer in [0, 90, 180, 270]

    # rotate and save image
    if angle == 90:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image, int(angle)


def preprocess_image_dir(
    input_dir: Path, output_dir: Path, image_list: List[str], args: argparse.Namespace
) -> Tuple[Dict[str, Any], bool]:
    """Preprocess images in input_dir and save them to output_dir.

    Args:
        input_dir (Path): Directory containing the a folder "images" with the images to preprocess.
        output_dir (Path): Directory to save the preprocessed images to.
        image_list (List[str]): List of image file names.
        args (argparse.Namespace): Arguments.

    Returns:
        Tuple[Dict[str, Any], bool]: Dictionary mapping image file names to rotation angles and
            whether all images have the same shape.
    """
    same_original_shapes = True
    prev_shape = None

    # log paths to debug
    logging.debug(f"Rescaling {input_dir / 'images'}")
    logging.debug(f"Saving to {output_dir / 'images'}")

    for image_fn in tqdm(image_list, desc=f"Rescaling {input_dir.name}", ncols=80):
        img_path = input_dir / "images" / image_fn
        image = cv2.imread(str(img_path))

        if prev_shape is not None:
            same_original_shapes &= prev_shape == image.shape

        prev_shape = image.shape

        # resize image
        if args.resize is not None:
            image = resize_image(image, args.resize)

        cv2.imwrite(str(output_dir / "images" / image_fn), image)

    # rotate image
    rotation_angles = {}
    n_rotated = 0
    n_total = len(image_list)

    same_rotated_shapes = True
    prev_shape = None

    if args.rotation_matching or args.rotation_wrapper:
        # log paths to debug
        logging.debug(f"Rotating {output_dir / 'images'}")
        logging.debug(f"Saving to {output_dir / 'images_rotated'}")

        features = output_dir / "sift_feat_rot.h5"
        matches = output_dir / "sift_matches_rot.h5"
        pairs = output_dir / "sift_pairs_rot.txt"
        #open(pairs, 'a').close()

        sift_config = {
            "model": {"name": "dog"},
            "options": {
                "first_octave": -1,
                "peak_threshold": 0.01,
            },
            "output": "sift_feat_rot",
            "preprocessing": {"grayscale": True, "resize_max": 1600},
        }
        NN_config ={ 
            'model': {
            'name': 'nearest_neighbor',
            'do_mutual_check': True,
            'ratio_threshold': 0.8,
            } 
        }
        extract_features.main(sift_config,input_dir / 'images',output_dir,image_list=image_list,feature_path=features)
        pairs_from_exhaustive.main(output=pairs, image_list=image_list)        
        match_features.main(NN_config,pairs=pairs,features=features,matches=matches)
        

        rotation_angles = estimate_rot_from_sift(features, matches)
        import tensorflow as tf

        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        weights = "ext_deps/dioad/weights/model-vit-ang-loss.h5"
        if args.kaggle:
            weights = f"/kaggle/input/imc-23-repo/IMC-2023/{weights}"

        deep_orientation = dioad.infer.Inference(load_model_path=weights)
        
        k = min(len(image_list),100)
        if k % 2==0:
            k-=1
        diff = np.zeros(k)
        for i, image_fn in tqdm(enumerate(image_list[:k]), desc=f"Dioad {input_dir.name}", ncols=80):
            img_path = output_dir / "images" / image_fn
            diff[i] = (deep_orientation.predict("vit", str(img_path)) - rotation_angles[image_fn]+180)%360 -180
        del deep_orientation
        gc.collect()
        np.set_printoptions(suppress=True, floatmode='fixed')
        print(diff)  
        diff = np.median(diff)

        for k in rotation_angles.keys():
            rotation_angles[image_fn]= (360 -diff -  rotation_angles[image_fn]) %360


        for image_fn in tqdm(image_list, desc=f"Rotating {input_dir.name}", ncols=80):
            img_path = output_dir / "images" / image_fn
            image = cv2.imread(str(img_path))
            
            angle = rotation_angles[image_fn]
            image, angle = get_rotated_image(image, angle)

            if prev_shape is not None:
                same_rotated_shapes &= prev_shape == image.shape

            prev_shape = image.shape

            if angle != 0:
                n_rotated += 1

            cv2.imwrite(str(output_dir / "images_rotated" / image_fn), image)


    logging.info(f"Rotated {n_rotated} of {n_total} images.")

    same_shape = same_rotated_shapes if args.rotation_wrapper else same_original_shapes

    logging.info(f"Images have same shapes: {same_shape}.")

    return rotation_angles, same_shape

