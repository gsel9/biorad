"""
Help: darshana.abeyrathna@uia.no
* Feature binarizing threshold: calc distr quantiles as initial esitmate
"""

import numpy as np
import pandas as pd

import pyximport;
pyximport.install(
    setup_args={"include_dirs":np.get_include()},
    reload_support=True
)


# Ensembles
ensemble_size = 1000

# Parameters for the Tsetlin Machine
T = 10 # Threshold
s = 3.0 # precision
number_of_clauses = 300
states = 100 # "Decisino states"

# Training configuration
epochs = 500

# Loading of training and test data
X = pd.read_csv('./no_filter_concat.csv', index_col=0).values
y = np.squeeze(pd.read_csv('./target_dfs.csv', index_col=0).values)

X[X > 5] = int(0)
X[X < 5] = int(1)

X = X.astype(np.int32)
y = y.astype(np.int32)


# Parameters of the pattern recognition problem
number_of_features = int(X.shape[1])
number_of_classes = 2

accuracy_training = np.zeros(ensemble_size)
accuracy_test = np.zeros(ensemble_size)


for ensemble in range(ensemble_size):
    print("ENSEMBLE", ensemble + 1)
    np.random.shuffle(X)

    X_training = X[:150, :] #data[:int(data.shape[0]*0.8),0:16] # Input features
    y_training = y[:150] #data[:int(data.shape[0]*0.8),16] # Target value

    X_test = X[150:, :] #data[int(data.shape[0]*0.8):,0:16] # Input features
    y_test = y[150:] #data[int(data.shape[0]*0.8):,16] # Target value

    # This is a multiclass variant of the Tsetlin Machine, capable of distinguishing between multiple classes
    tsetlin_machine = TsetlinMachine.TsetlinMachine(number_of_classes, number_of_clauses, number_of_features, states, s, T, boost_true_positive_feedback = 1)

    # Training of the Tsetlin Machine in batch mode. The Tsetlin Machine can also be trained online
    tsetlin_machine.fit(X_training, y_training, y_training.shape[0], epochs=epochs)

    # Some performacne statistics
    accuracy_test[ensemble] = tsetlin_machine.evaluate(X_test, y_test, y_test.shape[0])
    accuracy_training[ensemble] = tsetlin_machine.evaluate(X_training, y_training, y_training.shape[0])

    print("Average accuracy on test data: %.1f +/- %.1f" % (np.mean(100*accuracy_test[:ensemble+1]), 1.96*np.std(100*accuracy_test[:ensemble+1])/np.sqrt(ensemble+1)))
    print("Average accuracy on training data: %.1f +/- %.1f" % (np.mean(100*accuracy_training[:ensemble+1]), 1.96*np.std(100*accuracy_training[:ensemble+1])/np.sqrt(ensemble+1)))
    print()
