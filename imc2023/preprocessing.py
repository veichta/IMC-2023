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


def estimate_rot(name0,name1,features,matches):
    m, sc = get_matches(matches, name0, name1)
    if m.shape[0]<5:
        return 0, 0
    kp0, kp1 = get_keypoints(features, name0), get_keypoints(features, name1)
    m, sc = get_matches(matches, name0, name1)
    src_pts = np.array(kp0[m[:,0]], dtype=np.float32)
    dst_pts = np.array(kp1[m[:,1]], dtype=np.float32)
    pts = np.stack((src_pts, dst_pts))
    M, inliers = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC)
    U, _, Vt = np.linalg.svd(M[:,:2])
    R = U @ Vt
    angle = np.arctan2(R[1, 0], R[0, 0])
    #fig = plot_images([read_image(image_dir / name0), read_image(image_dir / name1)],titles=[name0, name1])
    #plot_matches(kp0[m[:,0]], kp1[m[:,1]], a=0.1)
    #plt.show()
    return np.rad2deg(angle), np.count_nonzero(inliers)
    

def propagate_rotation(current_image, accumulated_rotation,rotation_values,visited,max_spanning_tree,image_names):
    rotation_values[current_image] = accumulated_rotation
    neighbors = max_spanning_tree.neighbors(current_image)
    if visited[current_image]:
        print("backtrack")
        return
    visited[current_image]=1
    k = rotation_values[current_image]//90 % 4
    for neighbor in neighbors:
        if not visited[neighbor]:
            relative_rotation, _ =  estimate_rot(image_names[current_image], image_names[neighbor],features,matches)
            #show_img([current_image,neighbor],[round(rotation_values[current_image]/90) %4,round((rotation_values[current_image] + relative_rotation)/90) % 4],image_names)
            #plt.show()
            propagate_rotation(neighbor, accumulated_rotation + relative_rotation,rotation_values,visited,max_spanning_tree,image_names)

def rotation_from_sift(features,matches):
    image_names = sorted(list_h5_names(features))
    n = len(image_names)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in tqdm(range(n)):
        for j in range(i+1,n):
            rotation, weight = estimate_rot(image_names[i], image_names[j],features,matches)
            G.add_edge(i, j, weight=weight, rotation=rotation)
            G.add_edge(j, i, weight=weight, rotation=rotation)

    print(G)
    # Compute the maximum spanning tree
    max_spanning_tree = nx.maximum_spanning_tree(G)

    rotation_values = np.zeros(n)  # Initialize rotation values
    visited = np.zeros(n)
    propagate_rotation(0,0, rotation_values, visited, max_spanning_tree,image_names)

    offset90 = circmean(rotation_values,90)
    rotation_values -= offset90
    rotation_values = np.round(rotation_values/90)*90 % 360

    return dict(zip(image_names, rotation_values))

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
        

        rotation_angles = rotation_from_sift(features,matches)

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

