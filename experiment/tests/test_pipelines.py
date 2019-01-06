if __name__ == '__main__':
    # ToDos:
    # * Make Pipeline encompassing feature selection and classifcaiton.
    #   Remember to include StandardScaler at appropriate locations.
    # *
    # * Determine reasonable distributions for hparams (figure out how
    #   distribution boundaries are determined by sklearn).
    # *
    # TODO: Move to backend?
    import sys
    sys.path.append('./../../model_comparison')

    import os
    import backend
    import model_selection
    import comparison_frame

    from selector_configs import selectors
    from estimator_configs import classifiers

    from sklearn.datasets import mnist

    X =
    # Demo run:
    # * 97.80 % accuracy seems to be a fairly good score.
    #
    # TODO: Need seed + clf + selector name for unique ID to prelim results files.
    import model_selection

    from hyperopt import hp
    from hyperopt import tpe
    from hyperopt.pyll.base import scope

    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    # TEMP:
    from sklearn.feature_selection.univariate_selection import SelectPercentile, chi2
    from sklearn.ensemble import RandomForestClassifier

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )
    pipes_and_params = backend.formatting.pipelines_from_configs(
        selectors, classifiers
    )
    for pipe_label, estimator_items in pipes_and_params.items():
        pass
