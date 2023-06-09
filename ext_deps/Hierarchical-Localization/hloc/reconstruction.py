import argparse
import shutil
from typing import Optional, List, Dict, Any
import multiprocessing
from pathlib import Path
import pycolmap
import torch
from tqdm import tqdm

from . import logger
from .utils.database import COLMAPDatabase
from .triangulation import (
    import_features, import_matches, estimation_and_geometric_verification,
    OutputCapture, parse_option_args)

from .utils.io import get_features, get_matches
from .utils.guide import matches0_to_matches, sym_epipolar_distance_all, sym_homography_error_all
from .matchers.nearest_neighbor import NearestNeighbor


def create_empty_db(database_path: Path):
    if database_path.exists():
        logger.warning('The database already exists, deleting it.')
        database_path.unlink()
    logger.info('Creating an empty database...')
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    db.commit()
    db.close()


def import_images(image_dir: Path,
                  database_path: Path,
                  camera_mode: pycolmap.CameraMode,
                  image_list: Optional[List[str]] = None,
                  options: Optional[Dict[str, Any]] = None):
    logger.info('Importing images into the database...')
    if options is None:
        options = {}
    images = list(image_dir.iterdir())
    if len(images) == 0:
        raise IOError(f'No images found in {image_dir}.')
    with pycolmap.ostream():
        pycolmap.import_images(database_path, image_dir, camera_mode,
                               image_list=image_list or [],
                               options=options)


def get_image_ids(database_path: Path) -> Dict[str, int]:
    db = COLMAPDatabase.connect(database_path)
    images = {}
    for name, image_id in db.execute("SELECT name, image_id FROM images;"):
        images[name] = image_id
    db.close()
    return images


def run_reconstruction(sfm_dir: Path,
                       database_path: Path,
                       image_dir: Path,
                       verbose: bool = False,
                       options: Optional[Dict[str, Any]] = None,
                       ) -> pycolmap.Reconstruction:
    models_path = sfm_dir / 'models'
    models_path.mkdir(exist_ok=True, parents=True)
    logger.info('Running 3D reconstruction...')
    if options is None:
        options = {}
    options = {'num_threads': min(multiprocessing.cpu_count(), 16), **options}
    with OutputCapture(verbose):
        with pycolmap.ostream():
            reconstructions = pycolmap.incremental_mapping(
                database_path, image_dir, models_path, options=options)

    if len(reconstructions) == 0:
        logger.error('Could not reconstruct any model!')
        return None
    logger.info(f'Reconstructed {len(reconstructions)} model(s).')

    largest_index = None
    largest_num_images = 0
    for index, rec in reconstructions.items():
        num_images = rec.num_reg_images()
        if num_images > largest_num_images:
            largest_index = index
            largest_num_images = num_images
    assert largest_index is not None
    logger.info(f'Largest model is #{largest_index} '
                f'with {largest_num_images} images.')

    for filename in ['images.bin', 'cameras.bin', 'points3D.bin']:
        if (sfm_dir / filename).exists():
            (sfm_dir / filename).unlink()
        shutil.move(
            str(models_path / str(largest_index) / filename), str(sfm_dir))
    return reconstructions[largest_index]


def add_guided_matches(database, pairs_path, features, matches, max_error=10.0):
    image_ids = get_image_ids(database)
    with open(str(pairs_path), 'r') as f:
        pairs = [p.split() for p in f.readlines()]

    Model = NearestNeighbor
    model = Model({'distance_threshold': 2.0}).eval().to('cuda')

    db = COLMAPDatabase.connect(str(database))
    for (name0, name1) in tqdm(pairs):
        image_id0, image_id1 = image_ids[name0], image_ids[name1]
        tvg = db.read_two_view_geometry(image_id0, image_id1)

        m, s = get_matches(matches, name0, name1)

        if tvg['configuration_type'] == pycolmap.TwoViewGeometry.UNDEFINED:
            nm = m
            db.add_two_view_geometry(
                image_id0, image_id1, nm)
        else:
            data = {}
            data['keypoints0'], data['descriptors0'] = get_features(features, name0)
            data['keypoints1'], data['descriptors1'] = get_features(features, name1)

            data = {k : torch.from_numpy(v).float().cuda()[None] for k, v in data.items()}

            if tvg['configuration_type'] in [pycolmap.TwoViewGeometry.UNCALIBRATED,
                                             pycolmap.TwoViewGeometry.CALIBRATED]:
                F = torch.from_numpy(tvg['F']).float().cuda()
                if image_id1 > image_id0:
                    dist = sym_epipolar_distance_all(data['keypoints0'], data['keypoints1'], F)
                else:
                    dist = sym_epipolar_distance_all(data['keypoints1'], data['keypoints0'], F).transpose(-1, -2)
            else:
                H = torch.from_numpy(tvg['H']).float().cuda()
                if image_id1 > image_id0:
                    dist = sym_homography_error_all(data['keypoints0'], data['keypoints1'], H)
                else:
                    dist = sym_homography_error_all(data['keypoints1'], data['keypoints0'], H).transpose(-1, -2)
            
            data['bias'] = torch.where(dist < max_error, 0.0, -torch.inf)
            if image_id1 > image_id0:
                inl0, inl1 = tvg['inlier_matches'][:, 0], tvg['inlier_matches'][:, 1]
            else:
                inl0, inl1 = tvg['inlier_matches'][:, 1], tvg['inlier_matches'][:, 0]

            # avoid losing predicted inliers
            inl0, inl1 = inl0.astype(int), inl1.astype(int)
            data['bias'][:, inl0] = -torch.inf
            data['bias'][:, :, inl1] = -torch.inf
            data['bias'][:, inl0, inl1] = 4.0

            pred = model(data)
            m0 = pred['matches0'][0].cpu().short().numpy()

            nm = matches0_to_matches(m0)
            print(tvg['configuration_type'], image_id0, image_id1, nm.shape, tvg['inlier_matches'].shape)
            db.add_two_view_geometry(
                image_id0, image_id1, nm,
                tvg['F'], tvg['E'], tvg['H'], tvg['qvec'], tvg['tvec'])

    db.commit()
    db.close()


def main(sfm_dir: Path,
         image_dir: Path,
         pairs: Path,
         features: Path,
         matches: Path,
         camera_mode: pycolmap.CameraMode = pycolmap.CameraMode.AUTO,
         verbose: bool = False,
         skip_geometric_verification: bool = False,
         min_match_score: Optional[float] = None,
         image_list: Optional[List[str]] = None,
         image_options: Optional[Dict[str, Any]] = None,
         mapper_options: Optional[Dict[str, Any]] = None,
         guided_matching: bool = False,
         ) -> pycolmap.Reconstruction:

    assert features.exists(), features
    assert pairs.exists(), pairs
    assert matches.exists(), matches

    sfm_dir.mkdir(parents=True, exist_ok=True)
    database = sfm_dir / 'database.db'

    create_empty_db(database)
    import_images(image_dir, database, camera_mode, image_list, image_options)
    image_ids = get_image_ids(database)
    import_features(image_ids, database, features)
    import_matches(image_ids, database, pairs, matches,
                   min_match_score, skip_geometric_verification)
    if not skip_geometric_verification:
        estimation_and_geometric_verification(database, pairs, verbose)

    if guided_matching and not skip_geometric_verification:
        logger.info('Performing guided matching from 2-view geometry ...')
        add_guided_matches(database, pairs, features, matches)

    reconstruction = run_reconstruction(
        sfm_dir, database, image_dir, verbose, mapper_options)
    if reconstruction is not None:
        logger.info(f'Reconstruction statistics:\n{reconstruction.summary()}'
                    + f'\n\tnum_input_images = {len(image_ids)}')
    return reconstruction


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sfm_dir', type=Path, required=True)
    parser.add_argument('--image_dir', type=Path, required=True)

    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--matches', type=Path, required=True)

    parser.add_argument('--camera_mode', type=str, default="AUTO",
                        choices=list(pycolmap.CameraMode.__members__.keys()))
    parser.add_argument('--skip_geometric_verification', action='store_true')
    parser.add_argument('--min_match_score', type=float)
    parser.add_argument('--verbose', action='store_true')

    parser.add_argument('--image_options', nargs='+', default=[],
                        help='List of key=value from {}'.format(
                            pycolmap.ImageReaderOptions().todict()))
    parser.add_argument('--mapper_options', nargs='+', default=[],
                        help='List of key=value from {}'.format(
                            pycolmap.IncrementalMapperOptions().todict()))
    args = parser.parse_args().__dict__

    image_options = parse_option_args(
        args.pop("image_options"), pycolmap.ImageReaderOptions())
    mapper_options = parse_option_args(
        args.pop("mapper_options"), pycolmap.IncrementalMapperOptions())

    main(**args, image_options=image_options, mapper_options=mapper_options)
