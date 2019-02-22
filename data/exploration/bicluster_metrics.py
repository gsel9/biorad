
# -*- coding: utf-8 -*-
#
# bicluster_metrics.py
#


import numpy as np

from scipy.stats import spearmanr


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

    bic_avg = np.mean(bicluster.flatten())
    avg_rows = np.mean(bicluster, axis=1)[:, np.newaxis]
    avg_cols = np.mean(bicluster, axis=0)[np.newaxis, :]

    msr_values = (bicluster - (avg_rows - avg_cols + bic_avg)) ** 2

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

    bic_avg = np.mean(bicluster.flatten())
    avg_rows = np.mean(bicluster, axis=1)[:, np.newaxis]
    avg_cols = np.mean(bicluster, axis=0)[np.newaxis, :]

    msr_values = (bicluster - (avg_rows - avg_cols + bic_avg)) ** 2
    smsr_values = msr_values / (avg_rows ** 2 * avg_cols ** 2)

    return np.sum(smsr_values / (nrows * ncols))


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
        avg_cols_std = (avg_cols - np.mean(avg_cols)) / np.std(avg_cols)
    except:
        avg_cols_std = (avg_cols - np.mean(avg_cols))

    bic_std = _standardize_bicluster(bicluster)
    virt_error_values = np.absolute(bic_std - avg_cols_std)

    return np.sum(virt_error_values) / (nrows * ncols)


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


def avg_spearmans_rho(bicluster):
    """Compute the Average Spearman's Rho for a bicluster. A lower result
    close to 1 or -1 indicates a better bicluster.
    Ref.:
    Ayadi, W., Elloumi, M., & Hao, J. K. (2009). A biclustering algorithm based
    on a Bicluster Enumeration Tree: application to DNA microarray data.
    BioData mining, 2(1), 9.
    Args:
        bicluster (array-like): The bicluster data values.
    Returns:
        (float): The average spearman's rho score value.
    """

    nrows, ncols = np.shape(bicluster)

    row_corrs, _ = spearmanr(bicluster, axis=0)
    col_corrs, _ = spearmanr(bicluster, axis=1)

    if nrows <= 1 or ncols <= 1:
        return 0.0
    else:
        _genes, _samples = 0, 0
        for prev_row in range(nrows - 1):
            for next_row in range(prev_row + 1, nrows):

                corr, _ = spearmanr(bicluster[prev_row, :], bicluster[next_row, :])
                _genes += corr

        for prev_col in range(ncols - 1):
            for next_col in range(prev_col + 1, ncols):
                corr, _ = spearmanr(bicluster[:, prev_col], bicluster[:, next_col])
                _samples += corr

        genes = _genes / (nrows * (nrows - 1))
        samples = _samples / (ncols * (ncols - 1))

        return 2.0 * max(genes, samples)


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
    row_std[row_std == 0] = 1

    row_mean = np.mean(_bicluster, axis=0)

    return (_bicluster - row_mean) / row_std
