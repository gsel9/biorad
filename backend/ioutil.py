
"""
Note: joblib 0.12.2 restarts workers when a memory leak is detected.
"""

import re
import os
import csv
import nrrd
import shutil
import operator

import numpy as np
import pandas as pd
import scipy.io as sio

from pathlib import Path
from joblib import Parallel, delayed

from collections import OrderedDict
from multiprocessing import cpu_count


N_JOBS = cpu_count() - 1 if cpu_count() > 1 else cpu_count()


def _typecheck(item):
    # Return <int> if able to convert, else <str>.

    return int(item) if item.isdigit() else item


def swap_format(old_path, old_format, new_format, new_path=None):

    new_fname = os.path.basename(old_path).replace(old_format, new_format)

    if new_path is None:
        return os.path.join(os.path.dirname(old_path), new_fname)
    else:
        return os.path.join(new_path, new_fname)


def sample_num(sample):

    sample_ids = re.split('(\d+)', sample)
    for elem in sample_ids:
        if elem.isdigit():
            return int(elem)

    return None


def natural_keys(text):
    """
    """

    return [_typecheck(item) for item in re.split('(\d+)', text)]


def relative_paths(path_to_dir, target_format=None):
    """Produce a list of relative paths to all files in directory."""

    if target_format is None:
        raise ValueError('Must specify target format')

    if not os.path.isdir(path_to_dir):
        raise RuntimeError('Invalid path {}'.format(path_to_dir))

    file_names = sorted(os.listdir(path_to_dir))

    rel_paths = []
    for fname in file_names:

        rel_path = os.path.join(path_to_dir, fname)
        if os.path.isfile(rel_path) and rel_path.endswith(target_format):
            rel_paths.append(rel_path)

    return rel_paths


def matlab_to_nrrd(source_path, target_path, transform=None, **kwargs):
    """Converts MATLAB formatted images to NRRD format.

    Kwargs:
        path_mask (str):
        modality (str, {`mask`, `PET`, `CT`}):

    """

    global N_JOBS

    if os.path.isdir(source_path) and os.path.isdir(target_path):

        mat_rel_paths = relative_paths(source_path, target_format='.mat')
        for num, path_mat in enumerate(mat_rel_paths):

            image_data = sio.loadmat(path_mat)
            # Apply image transformation function.
            if transform is not None:
                image = transform(image_data[kwargs['modality']], **kwargs)
            else:
                image = image_data[kwargs['modality']]

            nrrd_path = swap_format(
                path_mat, old_format='.mat', new_format='.nrrd',
                new_path=target_path
            )
            nrrd.write(nrrd_path, image)
    else:
        raise RuntimeError('Unable to locate:\n{}\nor\n{}'
                           ''.format(source_path, target_path))

    return None


def sample_paths(path_image_dir, path_mask_dir, target_format=None):
    """Generate dictionary of locations to image and corresponding mask."""

    sample_paths = relative_paths(path_image_dir, target_format=target_format)
    mask_paths = relative_paths(path_mask_dir, target_format=target_format)

    samples = []
    for sample, mask in zip(sample_paths, mask_paths):
        samples.append(
            OrderedDict(
                Image=sample, Mask=mask, Patient=sample_num(sample), Reader=''
            )
        )
    return samples


def read_prelim_result(path_to_file):
    """Read temporary stored results."""

    results = pd.read_csv(path_to_file, index_col=False)

    return OrderedDict(zip(*(results.columns, results.values[0])))


def write_prelim_results(path_to_file, results):
    """Store results in temporary separate files to prevent write conflicts."""

    with open(path_to_file, 'w') as outfile:
        writer = csv.DictWriter(
            outfile, fieldnames=list(results.keys()), lineterminator='\n'
        )
        writer.writeheader()
        writer.writerow(results)

    return None


def write_final_results(path_to_file, results):
    """Write the total collection of results to disk."""

    if isinstance(results, pd.DataFrame):
        results.to_csv(path_to_file, columns=results.columns)
    else:
        data = pd.DataFrame([result for result in results])
        data.to_csv(path_to_file)

    return None


def setup_tempdir(tempdir, root=None):
    """Returns path and sets up directory if non-existent."""

    if root is None:
        root = os.getcwd()

    path_tempdir = os.path.join(root, tempdir)
    if not os.path.isdir(path_tempdir):
        os.mkdir(path_tempdir)

    return path_tempdir


def teardown_tempdir(path_to_dir):
    """Removes directory even if not empty."""

    shutil.rmtree(path_to_dir)

    return None


def _load_feature_batch(path_target_dir):

    fnames = os.listdir(path_target_dir)

    feature_sets = {}
    for num, fname in enumerate(fnames):
        feature_set[num] = pd.read_csv(fname, index_col=0)

    return feature_set


def load_feature_batches(path_ref_dir):

    target_dirs = [ref_dir for ref_dir in os.listdir(path_ref_dir)
        if not ref_dir.startswith('.') and not ref_dir.endswith('.csv')
    ]
    for target_dir in target_dirs:

        path_target_dir = os.path.join(path_ref_dir, target_dir)

        yield _load_feature_batch(path_target_dir)


if __name__ == '__main__':

    def ct_to_hu(image):
        #Convert CT intensity to HU.

        return image - 1024


    #matlab_to_nrrd(
    #    './../../data/images/ct_cropped_raw/',
    #    './../../data/images/ct_cropped_prep/',
    #    modality='CT', transform=None
    #)

    pet_samples = relative_paths('./../../data/images/pet_cropped', target_format='nrrd')
    ct_samples = relative_paths('./../../data/images/ct_cropped', target_format='nrrd')
    print(len(pet_samples), len(ct_samples))
