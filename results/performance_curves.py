import sys
sys.path.append('./../')


import numpy as np
import pandas as pd

from algorithms import feature_selection
from algorithms import classification
from utils import ioutil

from scipy import interp

from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import StratifiedKFold


def balanced_roc_auc(y_true, y_pred):

    return roc_auc_score(y_true, y_pred, average='weighted')


def fisher_pipe(random_state):

    pipe = Pipeline(
        [
            ('scaler', StandardScaler()),
            ('fisher_score', feature_selection.FisherScoreSelection(num_features=45)),
            ('dtree', classification.DecisionTreeClassifier(
                    class_weight='balanced',
                    criterion='gini',
                    max_depth=10,
                    max_features=None,
                    min_samples_leaf=0.21225229310675112,
                    random_state=random_state
                )
            )
        ]
    )
    return pipe


def gen_roc_curve():

    y = ioutil.load_target_to_ndarray(
        './../../removed_broken_slices_data/dfs_removed_broken_slices.csv'
    )
    X = ioutil.load_predictors_to_ndarray(
        './../../removed_broken_slices_data/all_features_removed_broken_slices.csv'
    )
    avg_tprs = []
    avg_aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    np.random.seed(0)
    random_states = np.random.choice(40, size=40)
    for random_state in random_states:

        model = fisher_pipe(random_state=random_state)
        folds = StratifiedKFold(n_splits=10, random_state=random_state)

        tprs = []
        aucs = []
        for train_idx, test_idx in folds.split(X, y):

            model.fit(X[train_idx, :], y[train_idx])
            probas_ = model.predict_proba(X[test_idx])

            # Compute ROC curve and area the curve.
            fpr, tpr, _ = roc_curve(y[test_idx], probas_[:, 1])
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)

        avg_tprs.append(np.mean(tprs))
        avg_aucs.append(np.mean(aucs))

    np.save('./results/roc_curve_aucs.npy', avg_aucs)
    np.save('./results/roc_curve_tprs.npy', avg_tprs)


def gen_validation_curve():

    np.random.seed(0)
    random_states = np.random.choice(40, size=40)

    num_features = np.arange(1, 52, 5)

    y = ioutil.load_target_to_ndarray(
        './../../removed_broken_slices_data/dfs_removed_broken_slices.csv'
    )
    X = ioutil.load_predictors_to_ndarray(
        './../../removed_broken_slices_data/all_features_removed_broken_slices.csv'
    )
    avg_train_scores, avg_test_scores = [], []
    std_train_scores, std_test_scores = [], []
    for random_state in random_states:
        fisher_train_scores, fisher_test_scores = validation_curve(
            fisher_pipe(random_state),
            X,
            y,
            param_name='fisher_score__num_features',
            param_range=num_features,
            cv=10,
            scoring=make_scorer(balanced_roc_auc)
        )
        avg_train_scores.append(np.mean(fisher_train_scores))
        avg_test_scores.append(np.mean(fisher_test_scores))
        std_train_scores.append(np.std(fisher_train_scores))
        std_test_scores.append(np.std(fisher_test_scores))

    np.save('./results/avg_validation_curve_train.npy', avg_train_scores)
    np.save('./results/avg_validation_curve_test.npy', avg_test_scores)
    np.save('./results/std_validation_curve_train.npy', std_train_scores)
    np.save('./results/std_validation_curve_test.npy', std_test_scores)


def gen_learning_curve(_):

    y = ioutil.load_target_to_ndarray(
        './../../original_images_data/dfs_original_images.csv'
    )
    X = ioutil.load_predictors_to_ndarray(
        './../../original_images_data/all_features_original_images.csv'
    )
    fisher_train_sizes, fisher_train_scores, fisher_test_scores = learning_curve(
        estimator=fisher_pipe(),
        X=X,
        y=y,
        train_sizes=np.linspace(0.1, 1.0, 50),
        cv=10,
        scoring=make_scorer(balanced_roc_auc),
        n_jobs=10
    )
    np.save(
        'fisher_learn_train_sizes.npy', fisher_train_sizes
    )
    np.save(
        'fisher_learn_train_scores.npy', fisher_train_scores
    )
    np.save(
        'fisher_learn_test_scores.npy', fisher_test_scores
    )
    dummy_train_sizes, dummy_train_scores, dummy_test_scores = learning_curve(
        estimator=dummy_pipe(),
        X=X,
        y=y,
        train_sizes=np.linspace(0.1, 1.0, 50),
        cv=10,
        scoring=make_scorer(balanced_roc_auc),
        n_jobs=10
    )
    np.save('dummy_learn_train_sizes.npy', dummy_train_sizes)
    np.save('dummy_learn_train_scores.npy', dummy_train_scores)
    np.save('dummy_learn_test_scores.npy', dummy_test_scores)


if __name__ == '__main__':
    gen_validation_curve()
    #gen_learning_curves()
