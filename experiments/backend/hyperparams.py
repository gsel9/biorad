# -*- coding: utf-8 -*-
#
# hyperparams.py
#


"""

Notes:
* How can determine optimal cache size?

HYPERPARAMETER DISTRIBUTIONS
- We want about an equal chance of ending up with a number of any order
  of magnitude within our range of interest.

"""

from hyperopt import hp


###############################################
##==== Hyperparameter generators ====##
###############################################


# Log-uniform between 1e-9 and 1e-4
#space['clf__alpha'] = hp.loguniform('clf__alpha', -9*np.log(10), -4*np.log(10))
# Random integer in 20:5:80
#space['clf__n_iter'] = 20 + 5 * hp.randint('clf__n_iter', 12)
# Random number between 50 and 100
#space['clf__class_weight'] = hp.choice('clf__class_weight', [None,]) #'balanced']),
# Discrete uniform distribution
#space['clf__n_estimators'] = scope.int(hp.quniform('clf__n_estimators', 20, 500, 5))
# Discrete uniform distribution
#space['clf__max_leaf_nodes'] = scope.int(hp.quniform('clf__max_leaf_nodes', 30, 150, 1))
# Discrete uniform distribution
#space['clf__min_samples_leaf'] = scope.int(hp.quniform('clf__min_samples_leaf', 20, 500, 5))



def hp_bool(name):
    """The search space for a boolean hyperparameter.

    Args:
        (name):

    """
    return hp.choice(name, [False, True])



######################################################
##==== Random forest hyperparameter generators ====##
######################################################


def _trees_criterion(name):
    """The search space for the function to measure the quality of a split in a
    random forest.

    Args:
        (name):

    """
    return hp.choice(name, ['gini', 'entropy'])


# Range a bit short?
def _trees_n_estimators(name):

    return scope.int(hp.qloguniform(name, np.log(9.5), np.log(3000.5), 1))


def _trees_max_features(name):
    return hp.pchoice(name, [
        (0.2, 'sqrt'),  # most common choice.
        (0.1, 'log2'),  # less common choice.
        (0.1, None),  # all features, less common choice.
        (0.6, hp.uniform(name + '.frac', 0., 1.))
    ])


def _trees_max_depth(name):
    return hp.pchoice(name, [
        (0.7, None),  # most common choice.
        # Try some shallow trees.
        (0.1, 2),
        (0.1, 3),
        (0.1, 4),
    ])


def _trees_min_samples_split(name):
    return 2


def _trees_min_samples_leaf(name):
    return hp.choice(name, [
        1,  # most common choice.
        scope.int(hp.qloguniform(name + '.gt1', np.log(1.5), np.log(50.5), 1))
    ])


# NOTE: Need only use bool space.
def _trees_bootstrap(name):

    return hp.choice(name, [True, False])


# ERROR:
# Random forest hyperparameters search space
def trees_param_space(
    name_func,
    n_estimators=None,
    max_features=None,
    max_depth=None,
    min_samples_split=None,
    min_samples_leaf=None,
    bootstrap=None,
    oob_score=False,
    n_jobs=-1,
    verbose=False
):
    """
    Generate trees ensemble hyperparameters search space.

    Args:
        name_func (function): Parameter label formatting function.

    """
    param_space = dict(
        n_estimators=(
            _trees_n_estimators(name_func('n_estimators'))
            if n_estimators is None else n_estimators
        ),
        max_features=(
            _trees_max_features(name_func('max_features'))
            if max_features is None else max_features
        ),
        max_depth=(
            _trees_max_depth(name_func('max_depth'))
            if max_depth is None else max_depth
        ),
        min_samples_split=(
            _trees_min_samples_split(name_func('min_samples_split'))
            if min_samples_split is None else min_samples_split
        ),
        min_samples_leaf=(
            _trees_min_samples_leaf(name_func('min_samples_leaf'))
            if min_samples_leaf is None else min_samples_leaf
        ),
        bootstrap=(
            _trees_bootstrap(name_func('bootstrap'))
            if bootstrap is None else bootstrap
        ),
        # Default settings defined according to descriptions found in
        # scikit-learn documentation.
        oob_score=oob_score,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    return param_space



##########################################################################
##==== Support Vector Machines hyperparameter generators ====##
#########################################################################


def _svm_gamma(name, n_features=1):
    """Generator of default gamma values for SVMs.

    This setting is based on the following rationales:
    1.  The gamma hyperparameter can be considered an amplifier of the
        original dot product or L2 norm.
    2.  The original dot product or L2 norm shall be normalized by
        the number of features first.

    """
    # -- making these non-conditional variables
    #    probably helps the GP algorithm generalize
    # assert n_features >= 1
    return hp.loguniform(
        name, np.log(1. / n_features * 1e-3), np.log(1. / n_features * 1e3)
    )


def _svm_degree(name):

    return hp.quniform(name, 1.5, 6.5, 1)


def _svm_tol(name):

    return hp.loguniform(name, np.log(1 + 1e-5), np.log(1 + 1e-2))


# ERROR:
# Support vector classifier hyperparameters search space.
# TODO:
# * Checkout how can make gamme independent of num features.
def svc_param_space(
    name_func,
    kernel,
    n_features=1,
    C=None,
    gamma=None,
    coef0=None,
    degree=None,
    shrinking=None,
    tol=None,
    class_weight='balanced',
    max_iter=-1,
    verbose=False,
    cache_size=300
):
    """
    Generate SVM hyperparamters search space.
    """
    # Degree only applies to the polynomial kernel.
    if kernel in ['linear', 'rbf', 'sigmoid']:
        _degree = 1
    else:
        _degree = (
            _svm_degree(name_func('degree')) if degree is None else degree
        )
    # Gamma only apllies to the RBF, polynomial and sigmoid kernels.
    if kernel in ['linear']:
        _gamma = 'auto'
    else:
        _gamma = (
            _svm_gamma(name_func('gamma'), n_features=n_features)
            if gamma is None else gamma
        )
        # TODO: Why?
        # Render gamma independent of n_features.
        _gamma = _gamma / n_features

    # Coef only applies to polynomial and sigmoid kernels.
    if kernel in ['linear', 'rbf']:
        _coef0 = 0.0
    elif coef0 is None:
        if kernel == 'poly':
            _coef0 = hp.pchoice(
                name_func('coef0'),
                [
                    (0.3, 0),
                    (0.7, _gamma * hp.uniform(name_func('coef0val'), 0, 10))
                ]
            )
        elif kernel == 'sigmoid':
            _coef0 = hp.pchoice(
                name_func('coef0'),
                [
                    (0.3, 0),
                    (0.7, _gamma * hp.uniform(name_func('coef0val'), -10, 10))
                ]
            )
        else:
            pass
    else:
        _coef0 = coef0

    param_space = dict(
        kernel=_kernel,
        gamma=_gamma,
        coef0=_coef0,
        degree=_degree,
        C=_svm_C(name_func('C')) if C is None else C,
        shrinking=(
            hp_bool(name_func('shrinking'))
            if shrinking is None else shrinking
        ),
        tol=_svm_tol(name_func('tol')) if tol is None else tol,
        # Default settings defined according to descriptions found in
        # scikit-learn documentation.
        verbose=verbose,
        class_weight=class_weight,
        cache_size=cache_size,
        max_iter=max_iter,
    )
    return param_space



###################################################################
##==== Naive Bayes classifiers hyperparameter generators ====##
###################################################################


def _gnb_var_smoothing(name):

    return hp.loguniform(
        name, np.log(1 + 1e-12), np.log(1 + 1e-7)
    )


# ERROR:
# Gaussian Naive Bayes hyperparameters search space
def gnb_param_space(name_func, priors=None, var_smoothing=None):

    param_space = dict(
        var_smoothing=(
            _gnb_var_smoothing(name_func('var_smoothing'))
            if var_smoothing is None else var_smoothing
        ),
        # Default settings defined according to descriptions found in
        # scikit-learn documentation.
        priors=priors
    )
    return param_space


###################################################################
##==== Logistic Regression hyperparameter generators ====##
###################################################################


def _logreg_penalty(name):

    return hp.pchoice(name, ['l2', 'l1'])


def _logreg_dual(name):
    # Dual formulation is only implemented for l2 penalty with liblinear
    # solver. Prefer dual = False when n_samples > n_features.

    return hp.choice(name, [False])


def _logreg_tol(name):
    """The search space for the stopping criterion tolerance parameter.

    Args:
        (name):

    """
    return hp.loguniform(name, np.log(1 + 1e-6), np.log(1 + 1e-2))


def _logreg_C(name):
    """The search space for the error term penalty parameter.

    Args:
        (name):

    """
    return hp.loguniform(name, np.log(1 + 1e-5), np.log(1e5))


def logreg_hparam_space(
    name_func,
    dual=False,
    solver='liblinear’',
    fit_intercept=True,
    intercept_scaling=1,
    class_weight=None,
    multi_class='ovr',
    max_iter=-1,
    verbose=0,
    warm_start=False,
    n_jobs=-1
):
    """
    Generate Logistic Regression hyperparamters search space.
    """
    param_space = dict(
        penalty=(
            _logreg_penalty(name_func('penalty'))
            if penalty is None else penalty
        ),
        tol=(
            _logreg_tol(name_func('tol'))
            if tol is None else tol
        ),
        C=(
            _logreg_C(name_func('C'))
            if C is None else C
        ),
        # Dual formulation is only implemented for l2 penalty with liblinear
        # solver. Prefer dual=False when n_samples > n_features.
        dual=dual,
        solver=solver,
        max_iter=max_iter,
        # Use OvR for binary problem.
        multi_class=multi_class,
        class_weight=class_weight,
        fit_intercept=fit_intercept,
        intercept_scaling=intercept_scaling,
        verbose=verbose,
        warm_start=warm_start,
        n_jobs=n_jobs
    )
    return param_space


###################################################################
##==== PLSRegression hyperparameter generators ====##
###################################################################


def _plsr_n_components(name, n_features=1):

    return scope.int(hp.quniform(name, 1, n_features, 5))


def _plsr_tol(name):

    return hp.loguniform(name, np.log(1 + 1e-6), np.log(1 + 1e-2))


def plsr_hparam_space(
    name_func,
    n_components=None,
    tol=None,
    n_features=1,
    scale=True,
    max_iter=1000,
    copy=True
):
    """
    Generate Logistic Regression hyperparamters search space.
    """
    param_space = dict(
        n_components=(
            _plsr_n_components(
                name_func('n_components'), n_features=n_features
            )
            if n_components is None else n_components
        ),
        tol=(
            _plsr_tol(name_func('tol'))
            if tol is None else tol
        ),
        max_iter=max_iter,
        scale=scale,
        copy=copy
    )
    return param_space



if __name__ == '__main__':
    # TODO: Plot hparam distributions for visual inspection.

    #hp_space = trees_param_space()
    #print(hp_space)
    def name_func():
        # Accepts a label from the pipeline estimator/transformer and merges
        # with the variable name in the hparam space function.
        pass
