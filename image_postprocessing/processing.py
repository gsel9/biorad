import os

from scipy.stats import mode
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BasePostprocessor(ABC):

    def __init__(self, data):

        self.data = data


class Postprocessor(BasePostprocessor):
    """Superclass for all preprocessors. Does nothing.
    """

    def __call__(self, images, targets):
        """The function being applied to the input images.
        """
        return images, targets

    def output_channels(self, input_channels):
        """The number of output channels as a function of input channels.
        """
        return input_channels

    def output_targets(self, input_targets):
        """The number of output channels as a function of input channels.
        """
        return input_targets


class WindowingProcessor(Postprocessor):
    """Used to set the dynamic range of an image.
    """

    def __init__(self, window_center, window_width, channel):
        self.window_center, self.window_width = window_center, window_width
        self.channel = channel

    def perform_windowing(self, image):

        image = image - self.window_center
        image[image < -self.window_width / 2] = -self.window_width / 2
        image[image > self.window_width / 2] = self.window_width / 2

        return image

    def __call__(self, images, targets):

        images = images.copy()
        images[..., self.channel] = self.perform_windowing(images[..., self.channel])

        return images, targets


class MultipleWindowsProcessor(WindowingProcessor):
    """Used to create multiple windows of the same channel.
    """

    def __init__(self, window_centers, window_widths, channel):
        self.window_centers = window_centers
        self.window_widths = window_widths
        self.channel = channel

    def generate_all_windows(self, images):
        channel = images[..., self.channel]
        new_channels = []
        for window_center, window_width in zip(self.window_centers, self.window_widths):
            self.window_center, self.window_width = window_center, window_width
            new_channel = self.perform_windowing(channel)
            new_channels.append(new_channel)

        return np.stack(new_channels, axis=-1)

    def __call__(self, images, targets):
        new_channels = self.generate_all_windows(images)

        # Replace current CT channel with all windowed versions
        images = np.delete(images, self.channel, axis=-1)
        images = np.concatenate((images, new_channels), axis=-1)
        return images, targets

    def output_channels(self, input_channels):
        return input_channels + len(self.window_widths) - 1


class PyRadiomicsFeatureProcessor(BasePostprocessor):
    """Process raw feature sets extracted with PyRadiomics.

    Args:
        path_to_features (array-like):
        indices (array-like):
        filter_type (str):
        error_dir (str)

    """


    # Nv: Number of voxels in ROI.
    # Ng: Numer of gray levels in image.
    hassan_transforms = {
        'firstorder_Entropy': lambda f, Nv: f * np.log(Nv),
        'glcm_DifferenceEntropy': lambda f, Ng: f / np.log(Ng ** 2),
        'glcm_JointEntropy': lambda f, Ng: f / np.log(Ng ** 2),
        'glcm_SumEntropy': lambda f, Ng: f / np.log(Ng ** 2),
        'glcm_Contrast': lambda f, Ng: f / (Ng ** 2),
        'glcm_DifferenceVariance': lambda f, Ng: f / (Ng ** 2),
        'glcm_SumAverage': lambda f, Ng: f / Ng,
        'glcm_DifferenceAverage': lambda f, Ng: f / Ng,
        'glrlm_GrayLevelNonUniformity': lambda f, Ng: f * Ng,
        'glrlm_HighGrayLevelRunEmphasis': lambda f, Ng: f / (Ng ** 2),
        'glrlm_ShortRunHighGrayLevelEmphasis': lambda f, Ng: f / (Ng ** 2),
        'ngtdm_Contrast': lambda f, Ng: f / Ng,
        'ngtdm_Complexity': lambda f, Ng: f / (Ng ** 3),
        'ngtdm_Strength': lambda f, Ng: f / (Ng ** 2),
    }

    def __init__(self, data):

        super().__init__(data)

    @classmethod
    def from_files(cls, path_to_files):
        """Read data from specified files."""
        datasets  = []
        for path_to_file in path_to_files:
            if os.path.isfile(path_to_file):
                dataset = pd.read_csv(path_to_file)
            else:
                raise RuntimeError(f'Cannot locate: {path_to_file}')

        return PyRadiomicsFeatureProcessor(datasets)

    def drop_redundant(self):
        """Remove redundant columns from PyRadiomics feature extraction."""
        for dataset in self.datasets:
            # Drop redundant columns from feature extraction.
            dataset.drop(cls.REDUNDANT_COLUMNS, axis=1, inplace=True)

        return self

    def set_index(self, index_values):
        """Assign index values to feature sets."""
        for dataset in self.datasets:
            dataset.index = index_values
        return self

    def filter_variance(self, paths_to_dropped, thresh):
        """Drop zero-variance features.

        """
        if not len(paths_to_dropped) == len(self.data):
            raise ValueError(f'Got {len(paths_to_dropped)} paths, but have '
                             f'have {len(self.data)} data sets.')
        for num, dataset in enumerate(self.datasets):
            to_drop = list(datasets.colums[dataset.var() <= thresh])
            dataset.drop(to_drop, axis=1, inplace=True)
            # Write redundant features to file.
            pd.Series(to_drop).to_csv(paths_to_dropped[num])

        return self

    def drop_missing(self, paths_to_dropped,):
        """Drop features with missing values.

        """
        if not len(paths_to_dropped) == len(self.data):
            raise ValueError(f'Got {len(paths_to_dropped)} paths, but have '
                             f'have {len(self.data)} data sets.')
        for num, dataset in enumerate(self.datasets):
            to_drop = dataset.columns[dataset.isnull().any()]
            dataset.drop(to_drop, axis=1, inplace=True)
            # Write features with missing values to file.
            pd.Series(to_drop).to_csv(paths_to_dropped[num])

        return self

    def apply_hassan_transform(self):
        """Apply Hassan transform to CT features.

        Definitions for feature transformas are given in Hassan et al
        2017/2018.

        """
        for dataset in self.datasets:
            for feature in dataset.columns:
                dataset[feature] = self.hassan_mappings[feature]
        return self

    @property
    def concatenated(self):

        return pd.concat(self.data, axis=1)

    @property
    def medians(self):

        return [dset.median().values for dset in self.data]

    @property
    def means(self):

        return [dset.mean().values for dset in self.data]

    @property
    def mins(self):

        return [dset.min().values for dset in self.data]

    @property
    def maxes(self):

        return [dset.max().values for dset in self.data]

    @property
    def vars(self):

        return [dset.var().values for dset in self.data]

    @property
    def stds(self):

        return [dset.std().values for dset in self.data]
