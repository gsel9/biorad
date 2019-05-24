from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import anderson
from scipy.stats import wilcoxon

import numpy as np


def test_normality(data, verbose=0, report=None, alpha=0.05):

    stat, p = shapiro(data)
    if p > alpha:
        if report is not None:
            report['shapiro'] = ('normal', p)
        if verbose > 0:
            print('Shapiro: Sample looks Gaussian (fail to reject H0)')
    else:
        if report is not None:
            report['shapiro'] = ('not normal', p)
        if verbose > 0:
            print('Shapiro: Sample does not look Gaussian (reject H0)')

    stat, p = normaltest(data)
    if p > alpha:
        if report is not None:
            report['K2'] = ('normal', p)
        if verbose > 0:
            print('K2: Sample looks Gaussian (fail to reject H0)')
    else:
        if report is not None:
            report['K2'] = ('not normal', p)
        if verbose > 0:
            print('K2: Sample does not look Gaussian (reject H0)')

    result = anderson(data)
    p = 0
    for i in range(len(result.critical_values)):
    	sl, cv = result.significance_level[i], result.critical_values[i]
    	if result.statistic < result.critical_values[i]:
            if report is not None:
                report['anderson'] = ('normal', p)
            if verbose > 0:
                print('Anderson: %.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
    	else:
            if report is not None:
                report['anderson'] = ('not normal', p)
            if verbose > 0:
                print('Anderson: %.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))

    if report is not None:
        return report


def wilcoxon_p_values(X, Y, return_p_value=False, alpha=0.05):

    def _check_failed_passed(label, same, diff, alpha, p_value):
        # Fail to Reject H0: Sample distributions are equal.
        # Reject H0: Sample distributions are not equal.
        if p_value > alpha:
            # Same distribution (fail to reject H0).
            same[label] = p_value
        else:
            # Different distribution (reject H0).
            diff[label] = p_value

        return same, diff

    (n_X_rows, n_X_cols), (n_Y_rows, n_Y_cols) = np.shape(X), np.shape(Y)

    same, diff, p_values = {}, {}, {}
    if n_X_rows != n_Y_rows:
        if n_Y_rows > n_X_rows:
            for col in X.columns:
                _, p_value = wilcoxon(
                    X.loc[:, col].values, Y.loc[X.index, col].values
                )
                p_values[col] = p_value
                same, diff = _check_failed_passed(col, same, diff, alpha, p_value)
        else:
            for col in X.columns:
                _, p_value = wilcoxon(
                    Y.loc[:, col].values, X.loc[Y.index, col].values
                )
                p_values[col] = p_value
                same, diff = _check_failed_passed(col, same, diff, alpha, p_value)
    else:
        for col in X.columns:
            _, p_value = wilcoxon(
                X.loc[:, col].values, Y.loc[:, col].values
            )
            p_values[col] = p_value
            same, diff = _check_failed_passed(col, same, diff, alpha, p_value)

    if not return_p_value:
        return same, diff

    return same, diff, p_values
