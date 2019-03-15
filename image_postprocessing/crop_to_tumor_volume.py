"""
Crop CT images to more effective visualize if cropped tumor regions contains
artifacts using 3D slicer software.
"""

import sys
sys.path.append('./../')

import os
import nrrd
import utils
import numpy as np


# ERROR:
def main(_):
    """Reads image, applies mask and writes cropped image to disk."""

    path_ct_imagedir = './../../../data_source/images/ct_nrrd/'
    path_masksdir = './../../../data_source/images/masks_nrrd/'
    path_cropped_dir = './../../../data_source/images/ct_cropped_nrrd'

    paths_items = utils.sample_paths(
        path_ct_imagedir, path_masksdir, target_format='nrrd'
    )
    for item in paths_items:

        image, _ = nrrd.read(item['Image'])
        mask, _ = nrrd.read(item['Mask'])

        cropped_image = image * mask
        cropped_path = os.path.join(path_cropped_dir, item['Image'].split('/')[-1])

        nrrd.write(cropped_path, cropped_image)


if __name__ == '__main__':
    main(None)
