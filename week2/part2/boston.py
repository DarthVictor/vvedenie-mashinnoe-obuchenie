# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 12:45:42 2017

@author: DarthVictor
"""
import pandas
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import scale
from sklearn.datasets import load_boston


#y = np.array(data_dirty['Class'])
#kf = KFold(n_splits=5, shuffle=True, random_state=42)
#
#X_1 = np.array(data_dirty[columns])
#def iteration_1(num_of_neighbors):
#    score = cross_val_score(KNeighborsClassifier(n_neighbors=num_of_neighbors), X_1, y, cv=kf, scoring='accuracy')
#    return [num_of_neighbors, np.mean(score)]
#
#results_1 = map(iteration_1, range(1, 51))


data_full = load_boston()
data = scale(data_full.data)
target = data_full.target
p_vector = np.linspace(1.0, 10.0, num=200)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
def iteration(p_i):
    regressor = KNeighborsRegressor(n_neighbors=5, weights='distance', p=p_i, metric='minkowski')
    score = cross_val_score(regressor, data, target, cv=kf, scoring='neg_mean_squared_error')
    return [p_i, np.mean(score)]

results = map(iteration, p_vector)
    
print(reduce(lambda a, b: a if a[1] > b[1] else b, results))