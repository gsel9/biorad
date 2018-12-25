import numpy as np


def bin_widths(image, nbins):
    """Compute bin width for image histogram."""

    return (np.max(image) - np.min(image)) / nbins
