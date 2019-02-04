import numpy as np
from sklearn import svm, datasets
from sklearn.model_selection import cross_val_score

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import InCondition

# Import SMAC-utilities
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

from sklearn.pipeline import Pipeline


from ReliefF import ReliefF
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

import pandas as pd

iris = datasets.load_iris()


#xls = pd.ExcelFile('X_endelig_squareroot.xlsx')
#data_raw_df = pd.read_excel(xls, sheet_name='tilbakefall', index_col=0)
#y = data_raw_df['Toklasser'].values
#X = data_raw_df.drop('Toklasser', 1)


X = pd.read_csv('./sqroot_concat.csv', index_col=0).values
y = np.squeeze(pd.read_csv('./target_dfs.csv', index_col=0).values)


cs = ConfigurationSpace()

#num_feats = UniformIntegerHyperparameter('selector__n_features_to_select', 1, 50)
#num_neighs = UniformIntegerHyperparameter('selector__n_neighbors', 5, 40)
#cs.add_hyperparameter(num_feats)
#cs.add_hyperparameter(num_neighs)

tol = UniformFloatHyperparameter('clf__tol', 1e-8, 1e-3)
n_comps = UniformIntegerHyperparameter('clf__n_components', 1, 20)
cs.add_hyperparameter(tol)
cs.add_hyperparameter(n_comps)

"""
# Build Configuration Space which defines all parameters and their ranges
cs = ConfigurationSpace()

# We define a few possible types of SVM-kernels and add them as "kernel" to our cs
kernel = CategoricalHyperparameter("kernel", ["linear", "rbf", "poly", "sigmoid"], default_value="poly")
cs.add_hyperparameter(kernel)


# There are some hyperparameters shared by all kernels
C = UniformFloatHyperparameter("C", 0.001, 1000.0, default_value=1.0)
shrinking = CategoricalHyperparameter("shrinking", ["true", "false"], default_value="true")
cs.add_hyperparameters([C, shrinking])

# Others are kernel-specific, so we can add conditions to limit the searchspace
degree = UniformIntegerHyperparameter("degree", 1, 5, default_value=3)     # Only used by kernel poly
coef0 = UniformFloatHyperparameter("coef0", 0.0, 10.0, default_value=0.0)  # poly, sigmoid
cs.add_hyperparameters([degree, coef0])
use_degree = InCondition(child=degree, parent=kernel, values=["poly"])
use_coef0 = InCondition(child=coef0, parent=kernel, values=["poly", "sigmoid"])
cs.add_conditions([use_degree, use_coef0])


# This also works for parameters that are a mix of categorical and values from a range of numbers
# For example, gamma can be either "auto" or a fixed float
gamma = CategoricalHyperparameter("gamma", ["auto", "value"], default_value="auto")  # only rbf, poly, sigmoid
gamma_value = UniformFloatHyperparameter("gamma_value", 0.0001, 8, default_value=1)
cs.add_hyperparameters([gamma, gamma_value])
# We only activate gamma_value if gamma is set to "value"
cs.add_condition(InCondition(child=gamma_value, parent=gamma, values=["value"]))
# And again we can restrict the use of gamma in general to the choice of the kernel
cs.add_condition(InCondition(child=gamma, parent=kernel, values=["rbf", "poly", "sigmoid"]))



"""

def svm_from_cfg(cfg):
    """ Creates a SVM based on a configuration and evaluates it on the
    iris-dataset using cross-validation.

    Parameters:
    -----------
    cfg: Configuration (ConfigSpace.ConfigurationSpace.Configuration)
        Configuration containing the parameters.
        Configurations are indexable!

    Returns:
    --------
    A crossvalidated mean score for the svm on the loaded data-set.
    """
    global iris

    # For deactivated parameters, the configuration stores None-values.
    # This is not accepted by the SVM, so we remove them.
    cfg = {k : cfg[k] for k in cfg if cfg[k]}
    # We translate boolean values:
    cfg["shrinking"] = True if cfg["shrinking"] == "true" else False
    # And for gamma, we set it to a fixed value or to "auto" (if used)
    if "gamma" in cfg:
        cfg["gamma"] = cfg["gamma_value"] if cfg["gamma"] == "value" else "auto"
        cfg.pop("gamma_value", None)  # Remove "gamma_value"

    clf = svm.SVC(**cfg, random_state=42)

    scores = cross_val_score(clf, iris.data, iris.target, cv=5)
    return 1 - np.mean(scores)  # Minimize!



def pipe_from_cfg(config):

    global X, y

    pipe = Pipeline(
        [
            ('scaler1', StandardScaler()),
            #('selector', ReliefF()),
            #('scaler2', StandardScaler()),
            ('clf', PLSRegression())
        ]
    )
    pipe.set_params(**config)
    scores = cross_val_score(pipe, X, y, cv=5, scoring='roc_auc', n_jobs=-1)

    return 1 - np.mean(scores)



# Scenario object
scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
                     "runcount-limit": 2,  # maximum function evaluations
                     "cs": cs,               # configuration space
                     "deterministic": "true",
                     'execdir': '.'
                     })


# Optimize, using a SMAC-object
print("Optimizing! Depending on your machine, this might take a few minutes.")
smac = SMAC(
    scenario=scenario,
    rng=np.random.RandomState(42),
    tae_runner=pipe_from_cfg
)
best_config = smac.optimize()

inc_value = pipe_from_cfg(incumbent)

#print("Optimized Value: %.2f" % (inc_value))
