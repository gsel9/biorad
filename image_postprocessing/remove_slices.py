import os
import re
import nrrd

import numpy as np
import pandas as pd

from scipy import stats
from operator import itemgetter

from ioutil import relative_paths
from image_graphics import show_ct_stack


def _make_new_filepath(path_to_newdir, path_to_oldfile):

    filename = os.path.basename(path_to_oldfile)
    return os.path.join(path_to_newdir, filename)


def _check_image(image, mask, display_stack_to_drop, num_neighbors, verbose=1):

    _, _, num_slices = np.shape(image)
    # Detect corrupted slices.
    to_drop = gl_window(image, num_neighbors)
    # Store metadata of operations.
    metadata = {
        'frac_corrupted_slices': np.size(to_drop) / num_slices,
        'num_corrupted_slices': np.size(to_drop),
        'num_orig_slices': num_slices,
        'include_image': 'no',
    }
    # Check if removing > 50% original GTV.
    frac_gtv_removed = np.sum(mask[:, :, to_drop]) / np.sum(mask)
    if frac_gtv_removed > 0.5:
        if verbose > 0:
            print('Discarded image!')

        metadata['num_slices_to_keep'] = 0
        return metadata, []

    if verbose > 0 and frac_gtv_removed > 0:
        print(f'Reducing GTV by {frac_gtv_removed * 100}')

    slices_to_keep = np.arange(num_slices, dtype=np.int32)
    if np.size(to_drop) > 0:
        slices_to_keep = np.delete(slices_to_keep, to_drop)

        if display_stack_to_drop:
            print('Discarding slices:')
            show_ct_stack(image[:, :, to_drop], slice_dim=2)

    if np.size(slices_to_keep) > 0:
        metadata['include_image'] = 'yes'
        metadata['num_slices_to_keep'] = np.size(slices_to_keep)

    return metadata, slices_to_keep


def _check_correspoding_image_mask(paths_to_image, paths_to_masks):

    for path_image, path_mask in zip(paths_to_image, paths_to_masks):
        image_fname = path_image.split('/')[-1]
        mask_fname = path_mask.split('/')[-1]

        image_num = re.findall(r'\d+', image_fname)
        mask_num = re.findall(r'\d+', mask_fname)

        if not image_num == mask_num:
            raise ValueError(f'Image {image_fname} and mask {mask_fname} does '
                             'not match')


def gl_window(image, num_neighbors):

    _, _, num_slices = np.shape(image)

    _image = np.copy(image)
    _image[image == 0] = np.nan
    # Learn threshold values from image.
    upper_thresh = np.nanmean(_image) + 3 * np.nanstd(_image)
    lower_thresh = np.nanmean(_image) - 3 * np.nanstd(_image)

    to_drop = []
    corr_slices = {}
    for slice_num in range(num_slices):
        image_slice = np.copy(_image[:, :, slice_num])

        dist_to_max = 0
        if np.nansum(image_slice > upper_thresh) > num_neighbors:
            dist_to_max = np.abs(np.nanmax(image_slice) - upper_thresh)

        dist_to_min = 0
        if np.nansum(image_slice < lower_thresh) > num_neighbors:
            dist_to_min = np.abs(np.nanmin(image_slice) - lower_thresh)

        # Record distance between slice extrema and threshold to rank which
        # slices to remove if needing to prioretize.
        if dist_to_min > 0 and dist_to_max > 0:
            corr_slices[slice_num] = max(dist_to_max, dist_to_min)

        # Sort slices by decreasing distance between extrema and threshold.
        sorted_slices = sorted(
            corr_slices.items(), key=itemgetter(1), reverse=True
        )
    if sorted_slices:
        to_drop, _ = zip(*sorted_slices)

    return to_drop


def drop_corrupted_slices(_):

    path_to_metadata = './removed_slices_metadata/windowing_mdata.csv'
    # NOTE: Must create new masks with the same size as reduced CT images.
    dirpath_new_ct = './../../data_source/images/ct_removed_broken_slices'
    dirpath_new_pet = './../../data_source/images/pet_removed_broken_slices'
    dirpath_new_ct_masks = './../../data_source/images/masks_removed_broken_slices_ct_size'
    dirpath_new_pet_masks = './../../data_source/images/masks_removed_broken_slices_pet_size'

    overwrite = True
    verbose = 1
    display_stack_to_drop = False
    display_stack_to_keep = False

    # This chi2 test is invalid when the observed or expected frequencies in
    # each category are too small. A typical rule is that all of the observed
    # and expected frequencies should be at least 5. Argue num neighbors >= 5
    # as is typicall criteria for Chi2 test.
    num_neighbors = 5

    paths_to_ct = relative_paths(
        './../../data_source/images/ct_nrrd', target_format='nrrd'
    )
    paths_to_pet = relative_paths(
        './../../data_source/images/pet_nrrd', target_format='nrrd'
    )
    paths_to_masks = relative_paths(
        './../../data_source/images/masks_nrrd', target_format='nrrd'
    )
    # NB: Check that CT/PET file names matches that of the masks.
    _check_correspoding_image_mask(paths_to_ct, paths_to_masks)
    _check_correspoding_image_mask(paths_to_pet, paths_to_masks)

    all_metadata = {}
    for num, path_to_ct in enumerate(paths_to_ct):
        ct_image, _ = nrrd.read(path_to_ct)
        pet_image, _ = nrrd.read(paths_to_pet[num])
        mask_image, _ = nrrd.read(paths_to_masks[num])
        # NB: If changing size of image used to identify corrupt slices, the
        # slice numbers will no longer match the original image.
        cropped_ct = ct_image * mask_image
        metadata, to_keep = _check_image(cropped_ct,
                                         mask_image,
                                         display_stack_to_drop,
                                         num_neighbors,
                                         verbose=verbose)
        # NOTE: Maintaining mask and PET data (not slicing).
        if np.size(to_keep) > 0:
            ct_stack = np.array(ct_image[:, :, to_keep], dtype=float)
            mask_ct_stack = np.array(mask_image[:, :, to_keep], dtype=int)

            pet_stack = np.array(pet_image[:, :, :], dtype=float)
            mask_pet_stack = np.array(mask_image[:, :, :], dtype=int)

            # Sanity check.
            assert np.shape(ct_stack) == np.shape(mask_ct_stack)
            assert np.shape(pet_stack) == np.shape(mask_pet_stack)

            # NOTE: Visualizing tumor region image only.
            if display_stack_to_keep:
                print('Keeping slices:')
                show_ct_stack(cropped_ct, slice_dim=2)

            write_image_to_nrrd(
                _make_new_filepath(dirpath_new_ct, path_to_ct),
                ct_stack,
                overwrite=overwrite
            )
            write_image_to_nrrd(
                _make_new_filepath(dirpath_new_ct_masks, paths_to_masks[num]),
                mask_ct_stack,
                overwrite=overwrite
            )
            write_image_to_nrrd(
                _make_new_filepath(dirpath_new_pet, paths_to_pet[num]),
                pet_stack,
                overwrite=overwrite
            )
            write_image_to_nrrd(
                _make_new_filepath(dirpath_new_pet_masks, paths_to_masks[num]),
                mask_pet_stack,
                overwrite=overwrite
            )
            all_metadata[path_to_ct] = metadata

    write_metadata(path_to_metadata, all_metadata, overwrite=overwrite)


def write_image_to_nrrd(path_to_file, image, overwrite=False, verbose=1):

    if verbose > 0:
        print(f'Writing to {path_to_file}')

    if overwrite:
        nrrd.write(path_to_file, image)
    else:
        if not os.path.isfile(path_to_file):
            nrrd.write(path_to_file, image)


def write_metadata(path_to_file, all_metadata, overwrite=False):

    df_metadata = pd.DataFrame(all_metadata)
    # Check if file already exists.
    if overwrite:
        df_metadata.to_csv(path_to_file)
    else:
        if not os.path.isfile(path_to_file):
            df_metadata.to_csv(path_to_file)


if __name__ == '__main__':
    drop_corrupted_slices(None)
