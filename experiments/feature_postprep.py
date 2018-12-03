import ioutil
import numpy as np
import pandas as pd
import logging


# NB ERROR:
def check_dtypes(features, valid_dtypes=None):
    # Encode dtypes

    if valid_dtypes is None:
        pass

def preprocessing_report():
    # Generate a report describing preprocessing steps
    # https://docs.python.org/3/howto/logging-cookbook.html
    pass


class PostProcessor:

    DROP_COLUMNS = [
        'Image', 'Mask', 'Reader', 'label', 'general'
    ]

    def __init__(self, path_to_features=None, verbose=0):

        self.verbose = verbose
        self.data = {
            num: pd.read_csv(fpath, index_col=0)
            for num, fpath in enumerate(path_to_features)
        }

        # NOTE:
        self.dropped_cols = {}

    def rename_columns(self, labels=None, add_extend=None):

        # Assign new column labels to each data set.
        if labels is not None:
            for feature_set in self.data.values():
                feature_set.columns = labels

        # Modify column labels with extension.
        if add_extend is not None:

            for data in self.data.values():
                new_labels = [
                    '{}_{}'.format(add_extend, col) for col in data.columns
                ]
                data.columns = new_labels

        return self


    def check_identifiers(self, id_col=None, target_id=None):

        if id_col is not None:

            for key, feature_set in self.data.items():

                feature_set.index = feature_set.loc[:, id_col]
                if id_col in feature_set.columns:
                    feature_set.drop(id_col, axis=1, inplace=True)
                    self.dropped_cols['{}_identifier'.format(key)] = id_col

        # Check matching identifier.
        if target_id is not None:

            for feature_set in self.data.values():

                if sum(feature_set.index != target_id) > 0:
                    raise RuntimeError('Different samples in feature set and '
                                       'reference samples!')
                else:
                    pass

        return self

    def check_features(self, steps='all'):

        if steps == 'all':

            self.filter_columns()
            self.impute_missing_values()
            #self.check_dtypes(features)

    def filter_columns(self, keys=None, columns=None):

        # Filter from all columns.
        if keys is None:
            for key, feature_set in self.data.items():

                # Drop default columns.
                if columns is None:

                    _cols = []
                    for label in self.DROP_COLUMNS:
                        _cols.extend(
                            list(feature_set.filter(regex=label).columns)
                        )
                    feature_set.drop(_cols, axis=1, inplace=True)
                    self.dropped_cols['{}_filtered'.format(key)] = _cols

                    if self.verbose > 0:
                        print('Dropped {} default columns'
                              ''.format(len(_cols)))
                else:
                    feature_set.drop(columns, axis=1, inplace=True)
                    self.dropped_cols['{}_filtered'.format(key)] = columns

                    if self.verbose > 0:
                        print('Dropped columns: {}'.format(len(columns)))

        # Filter from specified columns.
        else:
            if isinstance(keys, (list, tuple)):
                for key in keys:

                    # Drop default columns.
                    if columns is None:

                        _cols = []
                        for label in self.DROP_COLUMNS:
                            _cols.extend(
                                list(feature_set.filter(regex=label).columns)
                            )
                        self.data[key].drop(_cols, axis=1, inplace=True)
                        self.dropped_cols['{}_filtered'.format(key)] = _cols

                        if self.verbose > 0:
                            print('Dropped {} default columns'
                                  ''.format(len(_cols)))
                    else:
                        self.data[key].drop(columns, axis=1, inplace=True)
                        self.dropped_cols['{}_filtered'.format(key)] = columns

                        if self.verbose > 0:
                            print('Dropped columns: {}'.format(len(columns)))

            else:
                raise ValueError('Keys should be <list> or <tuple>, not {}'
                                 ''.format(type(keys)))

        return self

    def filter_constant_features(self):

        for key, data in self.data.items():

            filtered = data.loc[:, data.apply(pd.Series.nunique) != 1]
            const_cols = np.setdiff1d(data.columns, filtered.columns)
            data.drop(const_cols, axis=1, inplace=True)

            self.dropped_cols['{}_constant'.format(key)] = const_cols

            if self.verbose > 0:
                print('Dropped constant columns: {}'.format(len(const_cols)))

        return self

    def impute_missing_values(self, imputer=0, thresh=2):

        for feature_set in self.data.values():

            where_missing = np.where(feature_set.isnull())
            if isinstance(imputer, (int, float)):
                feature_set.iloc[where_missing] = imputer
            else:
                raise NotImplementerError('imputer not implemented')

            return self

    def drop_correlated(self, thresh=0.95):

        for key, data in self.data.items():

            if self.verbose > 0:
                _, num_org_feats = np.shape(data)

            # Create correlation matrix.
            corr_mat = data.corr().abs()

            # Select upper triangle of correlation matrix.
            upper = corr_mat.where(
                np.triu(np.ones(corr_mat.shape), k=1).astype(np.bool)
            )
            # Find index of feature columns with correlation > thresh.
            corr_cols = [
                col for col in upper.columns if any(upper[col] > thresh)
            ]
            data.drop(corr_cols, axis=1, inplace=True)

            if self.verbose > 0:
                if len(corr_cols) == 0:
                    print('Num dropped corr columns: {}'.format(0))
                else:
                    print('Num dropped corr columns: {}'.format(len(corr_cols)))

            self.dropped_cols['{}_correlated'.format(key)] = corr_cols
            self.dropped_cols['col_correlated_thresh'] = thresh

        return self


if __name__ == '__main__':

    import os

    path_to_dir = './../../data/prepped/discr_ct/'
    path_to_files = [
        os.path.join(path_to_dir, path_to_file)
        for path_to_file in os.listdir(path_to_dir)
    ]
    post_prep = PostProcessor(path_to_files, verbose=1)

    #for data in post_prep.data.values():
    #    print(np.shape(data))

    post_prep.drop_correlated(thresh=0.95)

    #for data in post_prep.data.values():
    #    print(np.shape(data))
