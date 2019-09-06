import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from model_comparison import model_comparison_experiment

from sklearn.datasets import make_classification
from sklearn.feature_selection import chi2
from sklearn.linear_model import RidgeClassifier
from scipy.stats import randint as sp_randint


def experiment():

    # The number cross-validation folds.
    CV = 2
    # The number of times each experiment is repeated with a different random seed.
    NUM_REPS = 2
    # The number of hyperparameter configurations to try evaluate.
    MAX_EVALS = 2
    # Name of the CSV file containing the experimental results.
    path_to_results = 'results.csv'

    # Define a series of models (example includes only one) wrapped in a
    # Pipeline.
    models = {
        'ridge': Pipeline([(f'{RidgeClassifier.__name__}', RidgeClassifier())])
    }

    # specify parameters and distributions to sample from
    hparams = {'ridge': {'RidgeClassifier__alpha': sp_randint(1, 11)}}

    X, y = make_classification(n_samples=50, n_features=4, n_classes=2)

    np.random.seed(seed=0)
    random_states = np.random.choice(1000, size=NUM_REPS)

    model_comparison_experiment(
        models=models,
        hparams=hparams,
        path_final_results=path_to_results,
        random_states=random_states,
        score_func=roc_auc_score,
        max_evals=MAX_EVALS,
        cv=CV,
        X=X,
        y=y
    )


if __name__ == '__main__':
    experiment()
