import os
import re
import nrrd

import numpy as np
import pandas as pd

from scipy import stats

from ioutil import relative_paths
from image_graphics import show_ct_stack


def crop_to_tumor_volume(image):

    coords = np.argwhere(image)
    x_min, y_min, z_min = coords.min(axis=0)
    x_max, y_max, z_max = coords.max(axis=0)

    return image[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]


def _check_image(image, upper_thresh, lower_thresh, display_stack_to_drop):

    # Detect corrupted slices.
    corrupted_slices = gl_window(image, upper_thresh, lower_thresh)
    """

    # Determine which slices to keep.
    _, _, num_slices = np.shape(image)
    slices_to_keep = np.arange(num_slices, dtype=np.int32)

    if np.size(corrupted_slices) > 0:
        slices_to_keep = np.delete(slices_to_keep, corrupted_slices)
        if display_stack_to_drop:
            print('Discarding slices:')
            show_ct_stack(image[:, :, corrupted_slices], slice_dim=2)

    # Store metadata of operations.
    metadata = {
        'frac_corrupted_slices': np.size(corrupted_slices) / num_slices,
        'num_corrupted_slices': np.size(corrupted_slices),
        'num_orig_slices': num_slices,
        'include_image': 'yes',
        'num_slices_to_keep': np.size(slices_to_keep)
    }
    if np.size(slices_to_keep) == 0:
        metadata['include_image'] = 'no'

    return metadata, slices_to_keep
    """


def _make_new_filepath(path_to_newdir, path_to_oldfile):

    filename = os.path.basename(path_to_oldfile)
    return os.path.join(path_to_newdir, filename)


def _check_correspoding_image_mask(paths_to_image, paths_to_masks):

    for path_image, path_mask in zip(paths_to_image, paths_to_masks):
        image_fname = path_image.split('/')[-1]
        mask_fname = path_mask.split('/')[-1]

        image_num = re.findall(r'\d+', image_fname)
        mask_num = re.findall(r'\d+', mask_fname)

        if not image_num == mask_num:
            raise ValueError(f'Image {image_fname} and mask {mask_fname} does '
                             'not match')


# ERROR:
def gl_window(image, lower_thresh, upper_thresh):
    # NOTE: Median is the statistic with the smallest STD across cropped CT
    # scans compared to mean and mode.

    _image = np.copy(image)
    # NOTE: Remove background from interfering with thresholds.
    _image[image == 0] = np.mean(image)

    lower_corrupt_slices = np.unique(np.where(_image < lower_thresh)[-1])
    upper_corrupt_slices = np.unique(np.where(_image > upper_thresh)[-1])

    corrupted_slices = np.unique(
        np.concatenate((lower_corrupt_slices, upper_corrupt_slices))
    )
    print(len(corrupted_slices) / image.shape[-1] * 100)
    return corrupted_slices


def drop_corrupted_slices(_):
    """

    Notes:
        * If automatically specifying a unique threshold per image or slice,
          e.g. histogram based, could risk that threshold is not affective if
          the image is very bright => almost all of the image is corrupted.
        * An attempt is made in cropping images to tumor volume avoiding a lot
          of sparse slices when determining the fraction of corrupted slices.

    """
    # TODO:
    # * Determine upper and lower thresh that results in removal of <= 50 % GTV.
    # * Include a threshold on the number of voxels > upper bound that must be
    #   surpassed in order to consider the voxels as artifacts.
    path_to_metadata = './removed_slices_metadata/windowing_ut1300_lt750.csv'
    dirpath_new_ct = './../../data_source/images/ct_slice_drop_ct1500_nrrd'
    dirpath_new_pet = './../../data_source/images/pet_slice_drop_ct1500_nrrd'
    dirpath_new_masks = './../../data_source/images/masks_slice_drop_ct1500_nrrd'
    overwrite = True
    upper_thresh = 1300
    lower_thresh = 750
    display_stack_to_drop = False
    display_stack_to_keep = False

    paths_to_ct = relative_paths(
        './../../data_source/images/ct_nrrd', target_format='nrrd'
    )
    paths_to_pet = relative_paths(
        './../../data_source/images/pet_nrrd', target_format='nrrd'
    )
    paths_to_masks = relative_paths(
        './../../data_source/images/masks_nrrd', target_format='nrrd'
    )
    # NOTE: Check that CT/PET file names matches that of the masks.
    _check_correspoding_image_mask(paths_to_ct, paths_to_masks)
    _check_correspoding_image_mask(paths_to_pet, paths_to_masks)

    # Remove slices from images and write to disk.
    all_metadata = {}
    for num, path_to_ct in enumerate(paths_to_ct):

        ct_image, _ = nrrd.read(path_to_ct)
        pet_image, _ = nrrd.read(paths_to_pet[num])
        mask_image, _ = nrrd.read(paths_to_masks[num])

        # Inspect only tumor region of CT image.
        cropped_ct = crop_to_tumor_volume(ct_image * mask_image)

        metadata, slices_to_keep = _check_image(
            cropped_ct, upper_thresh, lower_thresh, display_stack_to_drop
        )
    """
        if np.size(slices_to_keep) > 0.5:
            ct_stack = np.array(ct_image[:, :, slices_to_keep], dtype=float)
            pet_stack = np.array(pet_image[:, :, slices_to_keep], dtype=float)
            mask_stack = np.array(mask_image[:, :, slices_to_keep], dtype=int)

            # NOTE: Exclude image from data set if removing more than 50 % of
            # original ROI.
            #if np.sum(mask_stack) >= 0.5 * np.sum(mask_image):
            if np.sum(mask_stack) > 0:

                # Sanity check.
                assert np.shape(ct_stack) == np.shape(mask_stack)
                assert np.shape(pet_stack) == np.shape(ct_stack)

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
                    _make_new_filepath(dirpath_new_pet, paths_to_pet[num]),
                    pet_stack,
                    overwrite=overwrite
                )
                write_image_to_nrrd(
                    _make_new_filepath(dirpath_new_masks, paths_to_masks[num]),
                    mask_stack,
                    overwrite=overwrite
                )
            all_metadata[path_to_ct] = metadata

    write_metadata(path_to_metadata, all_metadata, overwrite=overwrite)
    """


def write_image_to_nrrd(path_to_file, image, overwrite=False):

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
    # TODO: Calc average volume removed from GTV in each image (can compare
    # radiomics feature volume between datasets).
    drop_corrupted_slices(None)
