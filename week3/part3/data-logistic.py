# -*- coding: utf-8 -*-
"""
@author: DarthVictor
"""

import numpy as np
import pandas as pd
import math
from sklearn.metrics import roc_auc_score
data_csv = pd.read_csv('data-logistic.csv', header=None)
X = np.array(data_csv[[1,2]])
y = np.array(data_csv[0])

def sigmoid(X, w):
    return 1 / (1 + np.exp(-np.dot(X, w)))

def gradient(X, y, C): # C = 0.0 - обычная, C=10.0 - l2-регуляризованная
        
    weights_length = 2
    weights_range = range(0, weights_length)
    i_length = len(X)
    i_range = range(0, i_length)
    prev_weights = np.zeros(weights_length)
    weights = np.zeros(weights_length)
    step = 0.1
    iter_num = 0
    max_iter = 10000
    delta = 10.0 ** -5
    current_dist = 10*delta
    while iter_num < max_iter and current_dist > delta:
        prev_weights = weights
        weights = np.zeros(weights_length)
        for weights_i in weights_range:
            acc = 0.0
            for i in i_range:
                acc = acc + (X[i][weights_i] * y[i] *
                  (1.0 - (1.0/
                      (1.0 + math.exp(
                      -y[i]*(X[i][0]*prev_weights[0] + X[i][1]*prev_weights[1]
                      )))
                  ))   
                )
            weights[weights_i] = prev_weights[weights_i] \
                       + (step*acc/i_length) \
                       - (step * prev_weights[weights_i] * C)
            
        current_dist = np.linalg.norm(weights-prev_weights)
        iter_num += 1
    print( iter_num, weights )
    return weights 

weights_non_reg = gradient(X, y, 0.0)
weights_l2 = gradient(X, y, 10.0)

score = roc_auc_score(y, sigmoid(X, weights_non_reg))
score_reg = roc_auc_score(y, sigmoid(X, weights_l2))
print (score)
print (score_reg)