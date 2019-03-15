import os
import nrrd

import numpy as np
import pandas as pd

from ioutil import relative_paths
from image_graphics import show_stack


def crop_to_tumor_volume(image):

    coords = np.argwhere(image)
    x_min, y_min, z_min = coords.min(axis=0)
    x_max, y_max, z_max = coords.max(axis=0)

    return image[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]


def _check_image(image, thresh, num_voxel_thresh=6):

    image = crop_to_tumor_volume(image)

    # If removing more than 50% of slices, features are shown to be affected.
    # Thus, the patient is dropped from the data set.
    _, _, num_slices = np.shape(image)

    # Detect and quantify corrupted slices.
    corrupted_slices = np.unique(np.where(image >= thresh)[-1])

    # Store metadata of operations.
    metadata = {
        'frac_corrupted_slices': np.size(corrupted_slices) / num_slices,
        'num_corrupted_slices': np.size(corrupted_slices),
        'num_orig_slices': num_slices
    }
    # Too many corrupted slices causes the image to be removed from the
    # data set.
    if metadata['frac_corrupted_slices'] > 0.5:
        metadata['include_image'] = 'no'
        return metadata, None

    # Correct for corrupted slices containing negligible number of arfitact
    # voxels.
    corr_corrupted_slices = [
        num for num in corrupted_slices
        if np.sum(image[:, :, num] > thresh) > num_voxel_thresh
    ]
    slices_to_keep = np.arange(num_slices)
    np.delete(slices_to_keep, corr_corrupted_slices)

    metadata['include_image'] = 'yes'
    metadata['num_slices_to_keep'] = np.size(slices_to_keep)

    return metadata, np.array(slices_to_keep, dtype=np.int32)


def _make_new_filepath(path_to_newdir, path_to_oldfile):

    filename = os.path.basename(path_to_oldfile)
    return os.path.join(path_to_newdir, filename)


def drop_corrupted_slices(_):
    """

    Notes:
        * If automatically specifying a unique threshold per image or slice,
          e.g. histogram based, could risk that threshold is not affective if
          the image is very bright => almost all of the image is corrupted.
        * An attempt is made in cropping images to tumor volume avoiding a lot
          of sparse slices when determining the fraction of corrupted slices.

    """

    # Setup:
    thresh = 1400
    display_stack = True
    path_to_metadata = './removed_slices_metadata/thres1400.csv'

    paths_to_ct = relative_paths(
        './../../data_source/images/ct_nrrd', target_format='nrrd'
    )
    paths_to_pet = relative_paths(
        './../../data_source/images/pet_nrrd', target_format='nrrd'
    )
    paths_to_masks = relative_paths(
        './../../data_source/images/masks_nrrd', target_format='nrrd'
    )
    dirpath_new_ct = './../../data_source/images/ct_dropped_slices_thresh1400_nrrd'
    dirpath_new_pet = './../../data_source/images/pet_dropped_slices_thresh1400_nrrd'
    dirpath_new_masks = './../../data_source/images/masks_dropped_slices_thresh1400_nrrd'

    # Remove slices from images and write to disk.
    all_metadata = {}
    for num, path_to_ct in enumerate(paths_to_ct):

        ct_image, _ = nrrd.read(path_to_ct)
        pet_image, _ = nrrd.read(paths_to_pet[num])
        mask_image, _ = nrrd.read(paths_to_masks[num])

        # Inspect only tumor region of CT image.
        cropped_ct = ct_image * mask_image
        metadata, slices_to_keep = _check_image(cropped_ct, thresh)

        if display_stack:
            show_stack(ct_image[:, :, slices_to_keep], slice_dim=2)

        # Exclude image from data set or write to disk.
        if ct_image is None:
            pass
        else:
            nrrd.write(
                _make_new_filepath(dirpath_new_ct, path_to_ct),
                ct_image[:, :, slices_to_keep]
            )
            nrrd.write(
                _make_new_filepath(dirpath_new_pet, paths_to_pet[num]),
                pet_image[:, :, slices_to_keep]
            )
            nrrd.write(
                _make_new_filepath(dirpath_new_masks, paths_to_masks[num]),
                mask_image[:, :, slices_to_keep]
            )
        all_metadata[path_to_ct] = metadata

    write_metadata(path_to_metadata, all_metadata)


def write_metadata(path_to_file, all_metadata):

    df_metadata = pd.DataFrame(all_metadata)
    # Check if file already exists.
    if os.path.isfile(path_to_file):
        pass
    else:
        df_metadata.to_csv(path_to_file)


if __name__ == '__main__':
    # TODO: Calc average volume removed from GTV in each image (can compare
    # radiomics feature volume between datasets).
    drop_corrupted_slices(None)
