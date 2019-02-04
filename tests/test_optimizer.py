import sys
sys.path.append('./..')
sys.path.append('./../../model_comparison')

import time
import backend

import numpy as np
import pandas as pd


from hyperopt import hp, tpe, fmin


def test_hyperopt():

    def objective(args):
        x,y = args
        f = x**2 - y**2
        return f

    param_space = [hp.uniform('x',-1,1), hp.uniform('y',-2,3)]

    best = fmin(objective,param_space,algo=tpe.suggest,max_evals=10)

    print(best)


test_hyperopt()
