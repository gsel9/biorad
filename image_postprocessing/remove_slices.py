# -*- coding: utf-8 -*-
#
# remove_slices.py
#

"""
"""

__author__ = 'Severin Langberg'
__contact__ = 'langberg91@gmail.com'


import os
import ast
import nrrd

import numpy as np

from utils import show_ct_stack
from utils import relative_paths
from utils import check_corresponding_image_mask


def _make_new_filepath(path_to_newdir, path_to_oldfile, new_format=None):

    filename = os.path.basename(path_to_oldfile)
    if new_format is not None:
        filename.replace(filename.split('.')[-1], new_format)

    return os.path.join(path_to_newdir, filename)


def gl_window(image, num_neighbors):

    _image = np.copy(image)
    _image[image == 0] = np.nan

    center = np.nanmedian(_image)
    upper_thresh = center + 100
    lower_thresh = center - 100

    _, _, num_slices = np.shape(image)

    corrupted = []
    for slice_num in range(num_slices):

        _image_slice = np.copy(_image[:, :, slice_num])
        image_slice = np.copy(_image_slice[_image_slice != np.nan])
        # Collect the slice number of corrupted slices.
        if np.sum(image_slice > upper_thresh) > num_neighbors:
            corrupted.append(slice_num)
        elif np.sum(image_slice < lower_thresh) > num_neighbors:
            corrupted.append(slice_num)

    return corrupted


def inspect_stacks(_):

    dirpath_slice_idx = './damaged_ct_slices'

    paths_to_ct = relative_paths(
        './../../data_source/images/ct_nrrd', target_format='nrrd'
    )
    paths_to_masks = relative_paths(
        './../../data_source/images/masks_nrrd', target_format='nrrd'
    )
    check_corresponding_image_mask(paths_to_ct, paths_to_masks)

    display_stack_to_drop = True
    display_stack_to_keep = False

    for num, path_to_ct in enumerate(paths_to_ct):

        image, _ = nrrd.read(path_to_ct)
        mask, _ = nrrd.read(paths_to_masks[num])
        cropped = image * mask

        print(f'Stack ID: {os.path.basename(path_to_ct)}')
        to_drop = gl_window(cropped, 1)

        frac_gtv_removed = np.sum(mask[:, :, to_drop]) / np.sum(mask)
        if frac_gtv_removed > 0.5:
            print(f'WARNING: Removing {frac_gtv_removed} of GTV!')

        if display_stack_to_drop:
            print('Discarding slices:')
            show_ct_stack(cropped[:, :, to_drop], slice_dim=2)

        if display_stack_to_keep:
            _, _, num_slices = np.shape(image)
            to_keep = np.arange(num_slices, dtype=np.int32)
            if np.size(to_drop) > 0:
                to_keep = np.delete(to_keep, to_drop)

            print('Keeping slices:')
            show_ct_stack(cropped[:, :, to_keep], slice_dim=2)

        path_to_file = _make_new_filepath(
            dirpath_slice_idx, path_to_ct, new_format='txt'
        )
        with open(path_to_file, 'w') as outfile:
            for slice_num in to_drop:
                outfile.write(f'{slice_num}\n')


def write_image_to_nrrd(path_to_file, image, overwrite=False, verbose=0):

    if verbose > 0:
        print(f'Writing to {path_to_file}')

    if overwrite:
        nrrd.write(path_to_file, image)
    else:
        if not os.path.isfile(path_to_file):
            nrrd.write(path_to_file, image)


def drop_slices_from_file(_):

    path_to_slices = './ct_slices_to_remove.txt'

    dirpath_new_ct = './../../data_source/images/ct_removed_broken_slices'
    dirpath_new_pet = './../../data_source/images/pet_removed_broken_slices'

    dirpath_new_ct_masks = './../../data_source/images/masks_removed_broken_slices_ct_size'
    dirpath_new_pet_masks = './../../data_source/images/masks_removed_broken_slices_pet_size'

    paths_to_ct = relative_paths(
        './../../data_source/images/ct_nrrd', target_format='nrrd'
    )
    paths_to_pet = relative_paths(
        './../../data_source/images/pet_nrrd', target_format='nrrd'
    )
    paths_to_masks = relative_paths(
        './../../data_source/images/masks_nrrd', target_format='nrrd'
    )
    check_corresponding_image_mask(paths_to_ct, paths_to_masks)
    check_corresponding_image_mask(paths_to_pet, paths_to_masks)

    with open(path_to_slices, 'r') as infile:
        to_drop = ast.literal_eval(infile.read())

    for num, path_to_ct in enumerate(paths_to_ct):

        ct_image, _ = nrrd.read(path_to_ct)
        pet_image, _ = nrrd.read(paths_to_pet[num])
        mask_image, _ = nrrd.read(paths_to_masks[num])

        patient_id, _ = os.path.basename(path_to_ct).split('.')
        if patient_id in to_drop.keys():

            _, _, num_slices = np.shape(ct_image)
            slices_to_keep = np.arange(num_slices, dtype=np.int32)
            start_slice, end_slice = to_drop[patient_id].split(':')
            try:
                slices_to_drop = np.arange(
                    int(start_slice), int(end_slice), dtype=int
                )
            except:
                slices_to_drop = np.arange(
                    int(start_slice), num_slices, dtype=int
                )
            slices_to_keep = np.delete(slices_to_keep, slices_to_drop)
            ct_image = np.array(ct_image[:, :, slices_to_keep], dtype=float)
            mask_ct_image = np.array(
                mask_image[:, :, slices_to_keep], dtype=int
            )
        else:
            mask_ct_image = mask_image

        frac_gtv_left = np.sum(mask_ct_image) / np.sum(mask_image)
        if frac_gtv_left < 0.5:
            pass
        else:
            write_image_to_nrrd(
                _make_new_filepath(dirpath_new_ct, path_to_ct),
                ct_image
            )
            write_image_to_nrrd(
                _make_new_filepath(dirpath_new_ct_masks, paths_to_masks[num]),
                mask_ct_image
            )
            write_image_to_nrrd(
                _make_new_filepath(dirpath_new_pet, paths_to_pet[num]),
                pet_image
            )
            write_image_to_nrrd(
                _make_new_filepath(dirpath_new_pet_masks, paths_to_masks[num]),
                mask_image
            )


if __name__ == '__main__':
    drop_slices_from_file(None)
