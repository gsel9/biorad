import os

import numpy as np
import pandas as pd


class PostProcessor:
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
