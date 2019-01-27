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

    self.hassan_mappings = {}

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

        self.data = None

    def produce(self, drop_redundant=True, drop_missing=True):
        """Apply a series of transformations to multiple feature sets.

        Kwargs:
            drop_redundant (bool): Removes features with zero variance.
            drop_missing (bool): Removes features missing values.

        Returns:
            (list): Processed feature sets.

        """
        self.data  = []
        for path_to_file in self.path_to_features:
            raw_data = pd.read_csv(path_to_file)

            # Drop redundant columns.
            _data = raw_data.filter(regex=self.filter_type)

            # Set index.
            if not np.array_equal(_data.index.values, self.indices):
                _data.index = self.indices
            # Drop redundant features.
            if drop_redundant:
                _data = self.drop_redundant(
                    self.get_filename(path_to_file), _data
                )
            # Drop missing features.
            if drop_missing:
                _data = self.drop_missing(
                    self.get_filename(path_to_file), _data
                )
            self.data.append(_data)

        return self

    def drop_redundant(self, filename, features):
        """Drop redundant features.

        """
        output = features.copy()

        redundant = output.columns[features.var() == 0.0].values
        if len(redundant) > 0:
            # Save redundant feature labels to disk.
            pd.Series(redundant).to_csv(
                os.path.join(self.error_dir, 'redundant_{}'.format(filename))
            )
            # Drop from feature set.
            output.drop(redundant, axis=1, inplace=True)

            if self.verbose > 0:
                print('Num redundant features: {}'.format(len(redundant)))

        return output

    def drop_missing(self, filename, features):
        """Drop features with missing values.

        """
        output = features.copy()

        missing = output.columns[features.isnull().any()].values
        if len(missing) > 0:
            # Save redundant feature labels to disk.
            pd.Series(missing).to_csv(
                os.path.join(self.error_dir, 'missing_{}'.format(filename))
            )
            # Drop from feature set.
            output.drop(missing, axis=1, inplace=True)

            if self.verbose > 0:
                print('Num missing features: {}'.format(len(missing)))

        return output

    @staticmethod
    def get_filename(path_to_file, file_format=None):
        """Extract filename from a path <str>.

        """
        fname = os.path.basename(path_to_file)

        # modify output file format.
        if not file_format:
            return fname

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
    def medians(self):

        return [dset.median().values for dset in self.data]

    @property
    def means(self):

        return [dset.mean().values for dset in self.data]

    @property
    def mins(self):

        [dset.min().values for dset in self.data]

    @property
    def maxes(self):

        [dset.max().values for dset in self.data]

    @property
    def stds(self):

        [dset.std().values for dset in self.data]
