import os
import re

from collections import OrderedDict

from natsort import natsorted



def relative_paths(path_to_dir, target_format=None):
    """Produce a list of relative paths to all files in directory."""

    if target_format is None:
        raise ValueError('Must specify target format')

    if not os.path.isdir(path_to_dir):
        raise RuntimeError('Invalid path {}'.format(path_to_dir))

    # Apply natural sorting to filenames.
    file_names = natsorted(os.listdir(path_to_dir))

    rel_paths = []
    for fname in file_names:

        rel_path = os.path.join(path_to_dir, fname)
        if os.path.isfile(rel_path) and rel_path.endswith(target_format):
            rel_paths.append(rel_path)

    return rel_paths


def _sample_num(sample):
    # Extracts patient ID.
    digit_in_str = re.findall(r'\d+', os.path.basename(sample))

    if len(digit_in_str) > 1:
        raise RuntimeError('Filename can only contain 1 number sequenze')

    return int(digit_in_str[0])


def sample_paths(path_image_dir, path_mask_dir, target_format=None):
    """Generate dictionary of locations to image and corresponding masks."""

    image_paths = relative_paths(path_image_dir, target_format=target_format)
    mask_paths = relative_paths(path_mask_dir, target_format=target_format)

    images_and_mask_paths = []
    for image_path in image_paths:

        # Match sample with mask by number.
        mask = None
        image_num = _sample_num(image_path)
        for mask_path in mask_paths:
            if _sample_num(mask_path) == image_num:
                images_and_mask_paths.append(
                    OrderedDict(
                        Image=image_path,
                        Mask=mask_path,
                        Patient=image_num,
                        Reader=''
                    )
                )
    return images_and_mask_paths
