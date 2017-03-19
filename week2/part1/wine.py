# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 12:45:42 2017

@author: DarthVictor
"""
import pandas
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale

data_dirty = pandas.read_csv('wine.csv')
print('>>>>>> ')
columns = [
'Alcohol', 'Malic acid','Ash', 'Alcalinity of ash', 'Magnesium', 
'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity',
'Hue', 'OD280/OD315 of diluted wines', 'Proline']

y = np.array(data_dirty['Class'])
kf = KFold(n_splits=5, shuffle=True, random_state=42)

X_1 = np.array(data_dirty[columns])
def iteration_1(num_of_neighbors):
    score = cross_val_score(KNeighborsClassifier(n_neighbors=num_of_neighbors), X_1, y, cv=kf, scoring='accuracy')
    return [num_of_neighbors, np.mean(score)]

results_1 = map(iteration_1, range(1, 51))

X_2 = scale(X_1)
def iteration_2(num_of_neighbors):
    score = cross_val_score(KNeighborsClassifier(n_neighbors=num_of_neighbors), X_2, y, cv=kf, scoring='accuracy')
    return [num_of_neighbors, np.mean(score)]

results_2 = map(iteration_2, range(1, 51))

print(reduce(lambda a, b: a if a[1] > b[1] else b, results_1))
print(reduce(lambda a, b: a if a[1] > b[1] else b, results_2))
