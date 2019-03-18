# -*- coding: utf-8 -*-
#
# utils.py
#

"""
"""

__author__ = 'Severin Langberg'
__contact__ = 'langberg91@gmail.com'


import nrrd
import matplotlib
matplotlib.use('TkAgg')
matplotlib.verbose.level = 'debug'

from PIL import Image
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt



def nrrd_to_ndarray(path_to_file, return_header=False):
    """

    Returns:
        (numpy.ndarray): Image stack as array.

    """
    data, header = nrrd.read(path_to_file)
    if return_header:
        return np.array(data, dtype=float), header
    else:
        return np.array(data, dtype=float)


def show_ct_stack(stack, slice_dim=0):
    """

    Notes:
        * vmin and vmax are set to 0 and 2e12, respectively.

    """

    class IndexTracker:
        def __init__(self, ax, x, slice_dim):

            self.ax = ax
            self.x = x
            self.slices = x.shape[slice_dim]
            self.slice_dim = slice_dim
            self.ind = self.slices // 2

            self.im = ax.imshow(
                self.x.take(
                    indices=self.ind,
                    axis=slice_dim,
                ),
                cmap=plt.cm.gray,
                vmin=0.0,
                vmax=2**12
            )
            self.update()

        def onscroll(self, event):
            if event.button == 'up':
                self.ind_add(1)
            else:
                self.ind_add(-1)

        def onclick(self, event):
            if event.key in ['right', 'down', 'd', 's']:
                self.ind_add(1)
            if event.key in ['left', 'up', 'a', 'w']:
                self.ind_add(-1)
            if event.key in [' ']:
                self.ind_add(50)


        def ind_add(self, count):
            self.ind = (self.ind + count) % self.slices
            self.update()

        def update(self):
            self.im.set_data(self.x.take(
                indices=self.ind,
                axis=self.slice_dim,
            ))
            ax.set_ylabel(f'slice {self.ind}')
            self.im.axes.figure.canvas.draw()

    fig, ax = plt.subplots(1, 1)
    tracker = IndexTracker(ax, stack, slice_dim)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    fig.canvas.mpl_connect('key_press_event', tracker.onclick)

    plt.show()


def img_dump(stack, dir: Path, base_name: str, slice_dim=0):
    """Take a stack and saves each stack as a image to disk.
    Args:
        stack: Patient CT-stack.
        dir (Path): Takes a filepath to store images.
        base_name (str): Name of the imagefiles.
        slice_dim: Slice dimension(0,1,2), 0 is default.
    """
    dir.mkdir(parents=True, exist_ok=True)
    for i in range(np.shape(stack)[slice_dim]):
        img = stack.take(
            indices=i,
            axis=slice_dim,
        )
        #Converts float32-values to 0 - 255 uint8 format
        formatted = (img * 255 /np.max(img)).astype('uint8')
        im = Image.fromarray(formatted)
        im.save(str(dir / f'{base_name}_{i}.jpeg'))


if __name__ == '__main__':
    # Demo run.
    path_to_stack = './../../data_source/images/ct_nrrd/P008CT.nrrd'
    path_to_mask = './../../data_source/images/masks_nrrd/P008mask.nrrd'

    image = nrrd_to_ndarray(path_to_stack)
    mask = nrrd_to_ndarray(path_to_mask)

    #cropped_image = image * mask

    plt.gray()

    show_stack(image)
    #img_dump(img_res, Path('./../lung_data/img'), 'stack_img', slice_dim=0)
