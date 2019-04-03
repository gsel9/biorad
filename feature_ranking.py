import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from mlxtend.evaluate import feature_importance_permutation

from algorithms import feature_selection
from algorithms import classification


def balanced_roc_auc(y_true, y_pred):

    return roc_auc_score(y_true, y_pred, average='weighted')


def fisher_pipe():

    pipe = Pipeline(
        [
            ('scaler', StandardScaler()),
            ('fisher_score', feature_selection.FisherScoreSelection(num_features=45)),
            ('dtree', classification.DecisionTreeClassifier(
                    class_weight='balanced',
                    criterion='gini',
                    min_samples_leaf=0.37755127,
                    random_state=0
                )
            )
        ]
    )
    return pipe


def run_permutation_importance(_):

    np.random.seed(0)
    random_states = np.random.choice(50, size=50)

    path_to_results = None

    X = None
    y = None

    model = fisher_pipe()

    importances = []
    for random_state in random_states:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )
        model.fit(X_train, y_train)
        imp_vals, _ = feature_importance_permutation(
            predict_method=model.predict,
            X=X_test,
            y=y_test,
            metric=make_scorer(balanced_roc_auc),
            num_rounds=100,
            seed=random_state
        )
        importances.append(imp_vals)

    np.save('permutation_importances.npy', importances)


if __name__ == '__main__':
    run_permutation_importance(None)
