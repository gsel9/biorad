# -*- coding: utf-8 -*-
#
# hyperparams.py
#


"""
Hyperparameter generators based on Hyperopt configurations and recommendations
by scikit-learn.

"""

import numpy as np

from hyperopt import hp
from hyperopt.pyll import scope


def hp_bool(name):
    """The search space for a boolean hyperparameter.

    Args:
        (name):

    """
    # Returns one of the alternatives.
    return hp.choice(name, (False, True))


def hp_num_features(name, max_num_features=1):
    """Returns a random integer in the range [0, upper].

    Args:
        name (str):
        max_num_features (int) The original size of the feature space.

    """
    # Cast to <int> according to hyperopt issue #253.
    return scope.int(hp.randint(name, max_num_features))


def hp_random_state(name):
    # Cast to <int> according to hyperopt issue #253.
    return scope.int(hp.randint(name, 1000))


######################################################
##==== Random forest hyperparameter generators ====##
######################################################


def _trees_n_estimators(name):
    # Equivalent to exp(uniform(low, high)).
    # Cast to <int> according to hyperopt issue #253.
    return scope.int(hp.qloguniform(name, np.log(9.5), np.log(3000.5), 1))

def _trees_criterion(name):
    # Returns one of the alternatives.
    return hp.choice(name, ['gini', 'entropy'])


def _trees_max_features(name):
    # Returns one of the `pchoice` alternatives. Probabilities render some
    # options selected more often than others. Most common is `sqrt`, while
    # `log2` or None (all features) are less common.
    return hp.pchoice(name, [
        (0.2, 'sqrt'),
        (0.1, 'log2'),
        (0.1, None),
        (0.6, hp.uniform(name + '.frac', 0.0, 1.0))
    ])


def _trees_max_depth(name):
    # Returns one of the `pchoice` alternatives. Most common is None
    # corresponding to an unpruned tree, while shallow trees are less common
    # with e.g. 2-4 levels.
    return hp.pchoice(name, [
        (0.7, None),
        (0.1, 2),
        (0.1, 3),
        (0.1, 4),
    ])


def _trees_min_samples_split(name):

    return int(2)


def _trees_min_samples_leaf(name):
    # Returns one of the alternatives. Most common choice is 1.
    # Casting to <int> denoting minimum number of samples required to be at a
    # leaf node rather than being interperated as a fraction if <float>.
    return hp.choice(name, [
        1,
        scope.int(hp.qloguniform(name + '.gt1', np.log(1.5), np.log(50.5), 1))
    ])


def trees_param_space(
    name_func,
    n_estimators=None,
    max_features=None,
    max_depth=None,
    criterion=None,
    min_samples_split=None,
    min_samples_leaf=None,
    bootstrap=None,
    random_state=None,
):
    """
    Generate trees ensemble hyperparameters search space.

    Args:
        name_func (function): Parameter label formatting function.
        n_estimators (int):
            Drawn from a exp(uniform(low, high)) distribution unless specified.
        criterion (str):
            Drawn randomly as `gini` or `entropy`.
        max_features (): Check scikit docs for common choices.
        max_depth (): Check scikit docs for common choices.
        min_samples_split (int):
            Defaults to two ... unless specified.
        min_samples_leaf ():
            Drawn randomly as one or from a exp(uniform(low, high))
            distribution.
        bootstrap (bool):
            Drawn randomly as true or false.
        oob_score (bool):
            Defults to false (check if recom at scikit docs).

    """
    param_space = {
        # n_estimators
        name_func('n_estimators'): _trees_n_estimators(
            name_func('n_estimators')
        ) if n_estimators is None else n_estimators,

        # criterion
        name_func('criterion'): _trees_criterion(name_func('criterion'))
        if criterion is None else criterion,

        # max_features
        name_func('max_features'): _trees_max_features(
            name_func('max_features')
        ) if max_features is None else max_features,

        # max_depth
        name_func('max_depth'): _trees_max_depth(name_func('max_depth'))
        if max_depth is None else max_depth,

        # min_samples_split
        name_func('min_samples_split'): _trees_min_samples_split(
            name_func('min_samples_split')
        ) if min_samples_split is None else min_samples_split,

        # min_samples_leaf
        name_func('min_samples_leaf'): _trees_min_samples_leaf(
            name_func('min_samples_leaf')
        ) if min_samples_leaf is None else min_samples_leaf,

        # bootstrap
        name_func('bootstrap'): hp_bool(name_func('bootstrap'))
        if bootstrap is None else bootstrap,

        # random_state
        name_func('random_state'): hp_random_state(name_func('random_state'))
        if random_state is None else random_state,

        # Predefined parameters.
        #name_func('oob_score'): oob_score,
        #name_func('n_jobs'): n_jobs,
        #name_func('verbose'): verbose,
    }
    return param_space



###############################################################
##==== Support Vector Machines hyperparameter generators ====##
###############################################################


def _svm_gamma(name, n_features=1):
    # Generator of default gamma values for SVMs. Equivalent to
    # exp(uniform(low, high)). The default hyperopt setting.
    #
    # This setting is based on the following rationales:
    # 1.  The gamma hyperparameter can be considered an amplifier of the
    #     original dot product or L2 norm.
    # 2.  The original dot product or L2 norm shall be normalized by
    #     the number of features first.
    # Args:
    #    name (str):
    #    n_features (int):
    #    -- making these non-conditional variables
    #       probably helps the GP algorithm generalize
    return hp.loguniform(
        name, np.log(1.0 / n_features * 1e-3), np.log(1.0 / n_features * 1e3)
    )


def _svm_degree(name):
    # Equivalent to round(uniform(low, high) / q) * q. The default hyperopt
    # setting.
    # Cast to <int> according to hyperopt issue #253.
    return scope.int(quniform(name, 1.5, 6.5, 1))


def _svm_tol(name):
    # Equivalent to exp(uniform(low, high)). The default hyperopt setting.
    return hp.loguniform(name, np.log(1e-5), np.log(1e-2))


def _svm_C(name):
    # Equivalent to exp(uniform(low, high)). The default hyperopt setting.
    return hp.loguniform(name, np.log(1e-5), np.log(1e5))


# TODO:
# * Checkout how can make gamme independent of num features.
def svc_param_space(
    name_func,
    kernel=None,
    gamma=None,
    degree=None,
    tol=None,
    C=None,
    shrinking=None,
    coef0=None,
    random_state=None,
    n_features=1,
):
    """
    Generate SVM hyperparamters search space.

    Args:
        name_func (): param...
        gamma (): param...
            Gamma only apllies to the RBF, polynomial and sigmoid kernels.
            Drawn from a exp(uniform(low, high)) distribution unless specified.
        degree (): param...
            Degree only applies to a polynomial kernel.
            Drawn from a round(uniform(low, high) / q) * q distribution
            unless specified.
        tol (): param...
            Drawn from a exp(uniform(low, high)) distribution unless specified.
        C (): param...
            Drawn from a exp(uniform(low, high)) distribution unless specified.
        shrinking (): param...
            Determined by boolean random choice unless specified.
        coef0 (): param...
            Coef only applies to polynomial and sigmoid kernels.
            Drawn from a gamma scaled uniform distribution depending on the
            specified kernel with 70 % probability, and 30 % probability of
            being defined equal to zero.
        kernel (): Defaults to RBF kernel.
        n_features (): Defaults to one feature.
        class_weight: Defaults to `balanced`.
        max_iter: Defaults to unlimited number of iterations for convergance.
        verbose: Defaults to false.
        cache_size: Defults to 500 by recommendation from scikit-learn and
            default settings in the `hyperopt` package.

    """
    # NB: Choose kernel initially due to dependence of other parameters.
    if kernel in ['linear', 'rbf', 'sigmoid']:
        _degree = 1
    else:
        _degree = (
            _svm_degree(name_func('degree')) if degree is None else degree
        ),
    if kernel in ['linear']:
        _gamma = 'auto'
    else:
        _gamma = (
            _svm_gamma(name_func('gamma'), n_features=n_features)
            if gamma is None else gamma
        )
        # Render gamma independent of n_features.
        _gamma = _gamma / n_features
    if kernel in ['linear', 'rbf']:
        _coef0 = 0.0
    elif coef0 is None:
        if kernel == 'poly':
            _coef0 = hp.pchoice(
                name_func('coef0'),
                [
                    (0.3, 0),
                    (0.7, _gamma * hp.uniform(name_func('coef0val'), 0., 10.))
                ]
            )
        elif kernel == 'sigmoid':
            _coef0 = hp.pchoice(
                name_func('coef0'),
                [
                    (0.3, 0),
                    (0.7, _gamma * hp.uniform(name_func('coef0val'), -10., 10.))
                ]
            )
        else:
            pass
    else:
        _coef0 = coef0
    # Generate hparam space.
    param_space = {
        name_func('kernel'): kernel,
        name_func('gamma'): _gamma,
        name_func('coef0'): _coef0,
        name_func('C'): _svm_C(name_func('C')) if C is None else C,
        name_func('shrinking'): hp_bool(name_func('shrinking'))
        if shrinking is None else shrinking,
        name_func('tol'): _svm_tol(name_func('tol')) if tol is None else tol,
        name_func('random_state'): hp_random_state(name_func('random_state'))
        if random_state is None else random_state,
    }
    return param_space



###############################################################
##==== Naive Bayes classifiers hyperparameter generators ====##
###############################################################


# ERROR: Invalid parameter var_smoothing for estimator GaussianNB(priors=None).
def _gnb_var_smoothing(name):

    # Equivalent to exp(uniform(low, high)).
    return hp.loguniform(name, np.log(1e-12), np.log(1e-7))


# ERROR: Invalid parameter var_smoothing for estimator GaussianNB(priors=None).
def gnb_param_space(name_func, priors=None, var_smoothing=None):
    """
    Generate Gaussian NB hyperparamters search space.

    Args:
        name_func (): param...
        priors ():
        var_smoothing ():
            Drawn randomly from exp(uniform(low, high)) distribution unless
            specified.

    """
    param_space = {
        #name_func('var_smoothing'): _gnb_var_smoothing(
        #    name_func('var_smoothing')
        #)
        #if var_smoothing is None else var_smoothing,
        name_func('priors'): priors
    }
    return param_space


###########################################################
##==== Logistic Regression hyperparameter generators ====##
###########################################################


def _logreg_penalty(name):
    # Returns one of the alternatives.
    return hp.choice(name, ['l1', 'l2'])


def _logreg_C(name):
    # Equivalent to exp(uniform(low, high)). Same settings as for SVM.
    return hp.loguniform(name, np.log(1e-5), np.log(1e5))


def _logreg_tol(name):
    # Equivalent to exp(uniform(low, high)). Shifted one tenth order of
    # magnitude down compared to SVM motivated by comparing logreg and SVM
    # default settings.
    return hp.loguniform(name, np.log(1e-6), np.log(1e-3))


def _logreg_inter_scaling(name):
    # From Hyperopt SVM settings.
    return hp.loguniform(name, np.log(1e-1), np.log(1e1))


def logreg_hparam_space(
    name_func,
    penalty=None,
    C=None,
    tol=None,
    random_state=None,
    fit_intercept=True,
    intercept_scaling=None,
):
    """
    Generate Logistic Regression hyperparamters search space.

    Args:
        penalty (str): Uniformly randomly selected as L1 or L2.
        C (float):
            Randomly drawn from exp(uniform(low, high)) distribution unless
            specified.
        tol (float):
            Randomly drawn from exp(uniform(low, high)) distribution unless
            specified.
        dual (bool):
            Defaults to false. Dual formulation is only implemented for L2
            penalty with liblinear solver. Prefer dual = False when
            n_samples > n_features according to scikit-learn documentation.
        multi_class (str):
            Use OvR for binary classification problem.

    """
    param_space = {
        name_func('penalty'): _logreg_penalty(name_func('penalty'))
        if penalty is None else penalty,
        name_func('C'): _logreg_C(name_func('C')) if C is None else C,
        name_func('tol'): _logreg_tol(name_func('tol')) if tol is None else tol,
        # Defaults to True in Hyperopt.
        name_func('fit_intercept'): fit_intercept,
        name_func('intercept_scaling'): _logreg_inter_scaling(
            name_func('intercept_scaling')
        ) if intercept_scaling is None else intercept_scaling,
        name_func('random_state'): hp_random_state(
            name_func('random_state') if random_state is None else random_state
        )
    }
    return param_space


######################################################
##==== PLS regression hyperparameter generators ====##
######################################################


def _plsr_n_components(name):
    # Assuming the PLSR impementation of scikit-learn is based on a similar
    # procedure to the PCA (see PLSR algorithm), the same `n_components` space
    # as for Hyperopt implementation of PCA can be transfered to PLSR.
    return 4 * scope.int(
        hp.qloguniform(name, low=np.log(0.51), high=np.log(30.5), q=1.0)
    )


def _plsr_tol(name):
    # Equivalent to exp(uniform(low, high)). Shifted according to default value
    # according to distributions of SVM and logreg tol params.
    return hp.loguniform(name, np.log(1e-8), np.log(1e-5))


def plsr_hparam_space(
    name_func,
    n_components=None,
    tol=None,
):
    """
    Generate PLS regression hyperparamters search space.
    """
    param_space = {
        name_func('n_components'): _plsr_n_components(
            name_func('n_components')
        ) if n_components is None else n_components,
        name_func('tol'): _plsr_tol(name_func('tol')) if tol is None else tol

    }
    return param_space


##############################################################
##==== Permutation Importance hyperparameter generators ====##
##############################################################


def permutation_hparam_space(name_func, num_features=None, model_hparams=None):
    """Combine parameters space for the wrapped model with hyperparameters of
    the permutation importance procedure.

    """

    param_space = {
        name_func('num_features'): hp_num_features(
            name_func('num_features'), max_num_features
        )
        if num_features is None else num_features
    }
    param_space.update(model_hparams)

    return hparam_space


############################################
##==== MRMR hyperparameter generators ====##
############################################


def wilcoxon_hparam_space(name_func, num_features=None):
    """

    Args:
        name_func ():
        num_features (int): The number of features select.

    """
    param_space = {
        name_func('num_features'): hp_num_features(
            name_func('num_features'), max_num_features
        )
        if num_features is None else num_features
    }
    return param_space


###############################################
##==== ReliefF hyperparameter generators ====##
###############################################


def _relieff_num_neighbors(name):
    # Equivalent to round(exp(uniform(low, high)) / q) * q emphasizing a
    # smaller number of neighbors.
    # Robnik-Sikonja and Kononenko (2003) showed that ReliefF’sestimates of
    # informative attribute are deteriorating with increasing number of nearest
    # neighbors in parity domain. Robnik-Sikonja and Kononenko also supports
    # Dalaka et al., 2000 with ten neighbors.
    return scope.int(hp.qloguniform(name, np.log(5), np.log(200), 1))


def relieff_hparam_space(
    name_func,
    num_neighbors=None,
    num_features=None,
    max_num_features=1
):
    """

    Args:
        name_func ():
        num_neighbors (int):
        num_features (int): The number of features select.
        max_num_features (int): Size of the original feature space.

    """
    param_space = {
        name_func('num_neighbors'): _relieff_num_neighbors(
            name_func('num_neighbors')
        )
        if num_neighbors is None else num_neighbors,
        name_func('num_features'): hp_num_features(
            name_func('num_features'), max_num_features
        )
        if num_features is None else num_features
    }
    return param_space


############################################
##==== MRMR hyperparameter generators ====##
############################################


def _mrmr_k(name):
    # Equivalent to round(exp(uniform(low, high)) / q) * q emphasizing a
    # smaller value of k.
    # Cast to <int> according to hyperopt issue #253.
    return scope.int(hp.qloguniform(name, np.log(2), np.log(12), 1))


def mrmr_hparam_space(
    name_func,
    k=None,
    num_features=None,
    max_num_features=1
):
    """

    Args:
        name_func ():
        k (int): Recommend a small integer between 3 and 10 [1].
        num_features (int): The number of features select.
        max_num_features (int): Size of the original feature space.

    References:
        [1]: A. Kraskov, H. St ̈ogbauer, and P. Grassberger.
             Estimating mutual information. Phys. Rev. E, 69(6):066138, 2004.

    """
    param_space = {
        name_func('k'): _mrmr_k(name_func('k')) if k is None else k,
        name_func('num_features'): hp_num_features(
            name_func('num_features'), max_num_features
        ) if num_features is None else num_features
    }
    return param_space



if __name__ == '__main__':
    pass
