import os

from scipy.stats import mode
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BasePreprocessor(ABC):

    @abstractmethod
    def __call__(self, images, targets):
        """The function being applied to the input images.
        """
        pass

    @abstractmethod
    def output_channels(self, input_channels):
        """The number of output channels as a function of input channels.
        """
        pass

    @abstractmethod
    def output_targets(self, input_targets):
        """The number of output channels as a function of input channels.
        """
        pass


class Preprocessor(BasePreprocessor):
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


class WindowingPreprocessor(Preprocessor):
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


class MultipleWindowsPreprocessor(WindowingPreprocessor):
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




class FeaturePostProcessor(Preprocessor):
    """Process raw feature sets extracted with PyRadiomics.

    Args:
        path_to_features (array-like):
        indices (array-like):
        filter_type (str):
        error_dir (str)

    """

    # Nv: Number of voxels in ROI.
    # Ng: Numer of gray levels in image.
    hassan_mappings = {
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

    def __init__(
        self,
        path_to_features,
        indices,
        filter_type,
        error_dir,
        verbose=1
    ):

        self.path_to_features = path_to_features
        self.indices = indices
        self.filter_type = filter_type
        self.error_dir = error_dir
        self.verbose = verbose

        self.data = self._read_data()

    def _read_data(self):
        # Read raw data.

        data  = []
        for path_to_file in self.path_to_features:

            if not os.path.isfile(path_to_file):
                raise RunetimeError('Unable to located: {}'
                                    ''.format(path_to_file))
            else:
                raw_data = pd.read_csv(path_to_file)

            # Drop redundant columns.
            _data = raw_data.filter(regex=self.filter_type)

            # Set index.
            if not np.array_equal(_data.index.values, self.indices):
                _data.index = self.indices

            data.append(_data)

        return data

    def process(self, drop_redundant=True, drop_missing=True, var_thresh=0):
        """Apply a series of transformations to multiple feature sets.

        Kwargs:
            drop_redundant (bool): Removes features with zero variance.
            drop_missing (bool): Removes features missing values.

        Returns:
            (list): Processed feature sets.

        """
        for num, dset in enumerate(self.data):
            # Drop redundant features.
            if drop_redundant:
                self.drop_redundant(
                    self.get_filename(self.path_to_features[num]),
                    dset,
                    var_thresh=var_thresh
                )
            # Drop missing features.
            if drop_missing:
                self.drop_missing(
                    self.get_filename(self.path_to_features[num]), dset
                )

        return self

    def drop_redundant(self, filename, features, var_thresh=0):
        """Drop redundant features.

        """
        redundant = features.columns[features.var() <= var_thresh].values
        if len(redundant) > 0:
            # Save redundant feature labels to disk.
            pd.Series(redundant).to_csv(
                os.path.join(self.error_dir, 'redundant_{}'.format(filename))
            )
            # Drop from feature set.
            features.drop(redundant, axis=1, inplace=True)

            if self.verbose > 0:
                print('Num redundant features: {}'.format(len(redundant)))

        return self

    def drop_missing(self, filename, features):
        """Drop features with missing values.

        """
        missing = features.columns[features.isnull().any()].values
        if len(missing) > 0:
            # Save redundant feature labels to disk.
            pd.Series(missing).to_csv(
                os.path.join(self.error_dir, 'missing_{}'.format(filename))
            )
            # Drop from feature set.
            features.drop(missing, axis=1, inplace=True)

            if self.verbose > 0:
                print('Num missing features: {}'.format(len(missing)))

        return features

    @staticmethod
    def get_filename(path_to_file, file_format=None):
        """Extract filename from a path <str>.

        """
        fname = os.path.basename(path_to_file)

        # modify output file format.
        if not file_format:
            return fname

    def to_file(self, path_to_dir, file_format=None, **kwargs):

        for num, features in enumerate(self.data):
            path_to_file = os.path.join(
                path_to_dir,
                self.get_filename(self.path_to_features[num], file_format)
            )
            features.to_csv(path_to_file, **kwargs)

        return self

    def hassan_transforms(self, features):
        """Apply Hassan transform to CT features.

        Definitions for feature transformas are given in Hassan et al
        2017/2018.

        Args:
            features ():

        Returns:
            ():

        """
        for feature_set in features:
            for feature in feature_set:
                feature_set[feature] = self.hassan_mappings[feature]

        return features

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
