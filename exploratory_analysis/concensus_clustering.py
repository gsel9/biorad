import warnings

import numpy as np
import pandas as pd


def mean_squared_residue(bicluster):
    """Compute Mean Squared Residue Score for a bicluster (MSR). A lower MSR
    score indicates a better bicluster.
    Ref.:
    Cheng, Y., & Church, G. M. (2000, August). Biclustering of expression data.
    In Ismb (Vol. 8, No. 2000, pp. 93-103).
    Args:
        bicluster (array-like): The bicluster data values.
    Returns:
        (float): The mean squared residue score.
    """

    nrows, ncols = np.shape(bicluster)

    bic_avg = np.mean(bicluster)
    avg_across_cols = np.mean(bicluster, axis=0)[np.newaxis, :]
    avg_across_rows = np.mean(bicluster, axis=1)[:, np.newaxis]
    avg_bic = avg_across_cols - avg_across_rows + bic_avg
    # Sanity check.
    assert np.shape(avg_bic) == np.shape(bicluster)

    msr_values = (bicluster - avg_bic) ** 2
    return np.sum(msr_values) / (nrows * ncols)


def scaled_mean_squared_residue(bicluster):
    """Compute Scaled Mean Squared Residue Score (SMSR) for a bicluster. A
    lower SMSR score indicates a better bicluster.
    Ref.:
    Mukhopadhyay, A., Maulik, U., & Bandyopadhyay, S. (2009). A novel coherence
    measure for discovering scaling biclusters from gene expression data.
    Journal of Bioinformatics and Computational Biology, 7(05), 853-868.
    Args:
        bicluster (array-like): The bicluster data values.
    Returns:
        (float): The scaled mean squared residue score.
    """

    nrows, ncols = np.shape(bicluster)

    bic_avg = np.mean(bicluster)
    avg_across_cols = np.mean(bicluster, axis=0)[np.newaxis, :]
    avg_across_rows = np.mean(bicluster, axis=1)[:, np.newaxis]
    avg_bic = avg_across_cols - avg_across_rows + bic_avg
    # Sanity check.
    assert np.shape(avg_bic) == np.shape(bicluster)

    msr_values = (bicluster - avg_bic) ** 2
    smsr_values = msr_values / (avg_across_cols ** 2 * avg_across_rows ** 2 + 1e-20)

    return np.sum(smsr_values / (nrows * ncols + 1e-20))


def virtual_error(bicluster):
    """Compute the Virtual Error (VE) of a bicluster. A lower VE score
    indicates a better bicluster.
    Ref.:
    Divina, F., Pontes, B., Giráldez, R., & Aguilar-Ruiz, J. S. (2012).
    An effective measure for assessing the quality of biclusters. Computers in
    biology and medicine, 42(2), 245-256.
    Args:
        bicluster (array-like): The bicluster data values.
    Returns:
        (float): The virtual error score.
    """

    nrows, ncols = np.shape(bicluster)

    avg_cols = np.mean(bicluster, axis=0)
    try:
        avg_cols_std = (avg_cols - np.mean(avg_cols)) / (np.std(avg_cols) + 1e-20)
    except:
        avg_cols_std = (avg_cols - np.mean(avg_cols))

    bic_std = _standardize_bicluster(bicluster)
    virt_error_values = np.absolute(bic_std - avg_cols_std)

    return np.sum(virt_error_values) / (nrows * ncols + 1e-20)


def transposed_virtual_error(bicluster):
    """Compute the Virtual Error of a transposed bicluster. A lower tranpsoed
    VE score indicates a better bicluster.
    Ref.:
    Pontes, B., Giráldez, R., & Aguilar-Ruiz, J. S. (2010, September).
    Measuring the Quality of Shifting and Scaling Patterns in Biclusters.
    In PRIB (pp. 242-252). Chicago.
    Args:
        bicluster (array-like): The bicluster data values.
    Returns:
        (float): The transposed virtual error score.
    """

    return virtual_error(np.transpose(bicluster))


def _standardize_bicluster(bicluster):
    """Standardize a bicluster by subtracting the mean and dividing by standard
    deviation.
    Ref.:
    Pontes, B., Girldez, R., & Aguilar-Ruiz, J. S. (2015). Quality measures
    for gene expression biclusters. PloS one, 10(3), e0115497.
    Note that UniBic synthetic data was generated with mean 0 and standard
    deviation 1, so it is already standardized.
    Args:
        bicluster (array-like): The bicluster data values.
    Returns:
        (float): The standardized bicluster.
    """

    _bicluster = np.copy(bicluster)

    row_std = np.std(_bicluster, axis=0)
    row_mean = np.mean(_bicluster, axis=0)

    return (_bicluster - row_mean) / (row_std + 1e-10)


def _external_metrics(indicators, nbiclusters, data):
    # Utility function for computing external metrics.

    row_idx, col_idx = indicators

    scores = {}
    for num in range(nbiclusters):

        _row_cluster = data[row_idx[num], :]
        cluster = _row_cluster[:, col_idx[num]]
        
        if np.any(cluster):
            scores[num] = {
                #'smr': mean_squared_residue(cluster),
                #'smsr': scaled_mean_squared_residue(cluster),
                #'vr': virtual_error(cluster),
                'tvr': transposed_virtual_error(cluster),
            }
        else:
            warnings.warn('Detected empty bicluster when calculating metrics.', Warning)
            scores[num] = {'tvr': np.nan}

    df_scores = pd.DataFrame(scores).T
    df_scores.index.name = 'ClusterID'

    return df_scores



class Biclusters:
    """Representation of a set of predicted biclusters."""

    def __init__(self, rows, cols, data):

        self.rows = rows
        self.cols = cols
        self.data = data

        # NOTE: Sets attributes.
        self._setup()

    @property
    def nbiclusters(self):

        return self._nbiclusters

    @nbiclusters.setter
    def nbiclusters(self, value):

        if np.shape(self.rows)[0] == np.shape(self.cols)[0]:
            self._nbiclusters = value
        else:
            raise RuntimeError('Sample clusters: {}, ref clusters {}'
                               ''.format(sample, ref))

    def _setup(self):

        self.nrows, self.ncols = np.shape(self.data)
        self.nbiclusters = np.shape(self.rows)[0]

        return self

    @property
    def indicators(self):
        """Determine coordiantes of row and column indicators
        for each bicluster.
        """

        row_idx, col_idx = [], []
        for cluster_num in range(self.nbiclusters):

            rows_bools = self.rows[cluster_num, :] != 0
            cols_bools = self.cols[cluster_num, :] != 0

            rows = [index for index, elt in enumerate(rows_bools) if elt]
            cols = [index for index, elt in enumerate(cols_bools) if elt]

            row_idx.append(rows), col_idx.append(cols)

        return row_idx, col_idx

    @property
    def stats(self):
        """Compute max, min and std from data points
        included in biclusters.
        """

        row_idx, col_idx = self.indicators
        data_size = np.size(self.data)

        stats = {}
        for num in range(self.nbiclusters):

            _row_cluster = self.data.values[row_idx[num], :]
            cluster = _row_cluster[:, col_idx[num]]
            if np.any(cluster):
                cluster_size = np.size(cluster)
                nrows, ncols = np.shape(cluster)

                stats[num + 1] = {
                    'max': np.max(cluster),
                    'min': np.min(cluster),
                    'std': np.std(cluster),
                    'nrows': nrows,
                    'ncols': ncols,
                    #'size': cluster_size,
                    'rel_size': cluster_size / data_size,
                    'zeros': int(np.count_nonzero(cluster==0))
                }
            else:
                pass

        df_stats = pd.DataFrame(stats).T
        df_stats.index.name = 'num'

        return df_stats

    @property
    def labels(self):
        """Assign row and column labels to biclusters."""

        genes = np.array(self.data.columns, dtype=object)
        cpgs =  np.array(self.data.index, dtype=object)

        row_idx, col_idx = self.indicators

        row_labels, col_labels = [], []
        for num in range(self.nbiclusters):
            row_labels.append(cpgs[row_idx[num]])
            col_labels.append(genes[col_idx[num]])

        return row_labels, col_labels

    @property
    def external_metrics(self):
        """Compute external evaluation metrics for each bicluster."""

        return _external_metrics(
            self.indicators, self.nbiclusters, self.data
        )

    def to_disk(self, file_name, parent='./../predictions/biclusters/'):
        """Generate txt files containing row and column indicators for
        detected biclusters associated with different datasets.
        Args:
            file_name (str): Name of file.
        """

        row_labels, col_labels = self.labels
        with open(os.path.join(parent, file_name), 'w') as outfile:
            for num in range(self.nbiclusters):
                outfile.write('bicluster_{0}\n'.format(num))
                outfile.write('{0}\n'.format(row_labels[num]))
                outfile.write('{0}\n'.format(col_labels[num]))

        return self
