# -*- coding: utf-8 -*-
#
# utils.py
#

"""
Radiomics feature extraction utility functions.
"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


import nrrd

import numpy as np


def bin_width(path_to_images, num_bins):
    """Computes average bin width for a stack of images resulting in a target
    number of bins.

    Args:
        path_to_images (str): Referance to location of images.
        num_bins (int): Suggested number of bins.

    Return:
        (float): Average bin width.

    """
    image_max, image_min = [], []
    for num, sample_items in enumerate(path_to_images):
        # Fetch sample items.
        image_path, mask_path, _, _ = sample_items.values()
        # Read mask and image data.
        mask, _ = nrrd.read(mask_path)
        raw_image, _ = nrrd.read(image_path)
        # Crop the image.
        cropped = raw_image * mask

        image_max.append(np.max(cropped)), image_min.append(np.min(cropped))
    # NOTE: Less expensive to store min/max values and apply operations later
    # compared to ceil and avg for each sample.
    return (np.mean(image_max) - np.mean(image_min)) / num_bins
