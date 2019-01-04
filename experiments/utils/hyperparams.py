# -*- coding: utf-8 -*-
#
# hyperparams.py
#


"""
Specify the distribtuion associated with each hyperparameter.

To Dos:
* Collect references to recommended parameter distributions.

"""


from hyperopt import hp


# HYPERPARAMETER DISTRIBUTIONS
# - We want about an equal chance of ending up with a number of any order
# of magnitude within our range of interest.
n_estimators = stats.expon(scale=100)
max_depth = stats.randint(1, 40)



# Parameter search space
space = {}

# Random number between 50 and 100
space['kbest__percentile'] = hp.uniform('kbest__percentile', 50, 100)

# Random number between 0 and 1
space['clf__l1_ratio'] = hp.uniform('clf__l1_ratio', 0.0, 1.0)

# Log-uniform between 1e-9 and 1e-4
space['clf__alpha'] = hp.loguniform('clf__alpha', -9*np.log(10), -4*np.log(10))

# Random integer in 20:5:80
space['clf__n_iter'] = 20 + 5 * hp.randint('clf__n_iter', 12)

# Random number between 50 and 100
space['clf__class_weight'] = hp.choice('clf__class_weight', [None,]) #'balanced']),

# Discrete uniform distribution
space['clf__n_estimators'] = scope.int(hp.quniform('clf__clf__n_estimators', 20, 500, 5))

# Discrete uniform distribution
space['clf__max_leaf_nodes'] = scope.int(hp.quniform('clf__max_leaf_nodes', 30, 150, 1))

# Discrete uniform distribution
space['clf__min_samples_leaf'] = scope.int(hp.quniform('clf__min_samples_leaf', 20, 500, 5))


estimator_hparams = {
    # Classification hyperparameters.
    'rf': {
        # From scikit-learn example (typically, many estimators = good).
        # From blog:  uniform
        'n_estimators': stats.expon(scale=100),
        # (typically, deep = good)
        # From blog: log-normal
        'max_depth': stats.randint(1, 40),
        # Has somethin gto do with feature selection? Should thus be skipped.
        'max_features': stats.randint(1, 11),
        # From blog: normal distribution
        'min_samples_split': stats.randint(2, 11),
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy'],
        'class_weight': ['balanced'],
    },
    'svc': {
        # class_weight: balanced by default.
        # Limiting the number of unique values (size=100) to ensure a
        # certain degree of diversity in the hparam values.
        # Small tolerance typically good, but want to checkout a few
        # alternatives.
        'tol': stats.reciprocal(size=10),
        # From scikit-learn docs.
        'C': stats.expon(scale=100),
        # From scikit-learn docs.
        'gamma': stats.expon(scale=.1),
        'kernel': ['linear', 'rbf', 'ploy'],
        'degree': [2, 3],
        'max_iter': [2000]
    },
    'logreg': {
        'tol': stats.reciprocal(size=10),
        'C': stats.expon(scale=100),
        'class_weight': ['balanced'],
        'penalty': ['l1', 'l2'],
        'max_iter': [2000],
        'solver': ['liblinear'],
    },
    'gnb': {
        # DFS:
        'priors': [[0.677, 0.323]]
        # LRR:
        # 'priors'. [[0.75, ]]
    },
    'plsr': {
        'tol': stats.reciprocal(size=10),
        'n_components': stats.expon(scale=100),
        'max_iter': [2000]
    },



    selector_hparams = {
        # Feature selection hyperparameters.
        'permutation': {
            'num_rounds': [100]
        },
        'wlcx': {
            'thresh': [0.05]
        },
        'relieff': {
            'num_neighbors': stats.expon(scale=100),
            'num_features': stats.expon(scale=100)
        },
        'mrmr': {
            'num_features': ['auto'],
            # See paper
            'k': [3, 5, 7]
        }
    }
