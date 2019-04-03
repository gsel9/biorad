import numpy as np
from scipy import interp
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold


def main(_):

    X =
    y =

    model =

    # Setup:
    folds = StratifiedKFold(n_splits=10, random_state=0)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for num, (train, test) in enumerate(folds.split(X, y)):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

    np.save('X_dtree_')
    np.save()
    np.save()

if __name__ == '__main__':
    main(None)
