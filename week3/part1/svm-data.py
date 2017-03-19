# -*- coding: utf-8 -*-
"""
@author: DarthVictor
"""
import pandas
import numpy as np
from sklearn.svm import SVC


data_csv = pandas.read_csv('svm-data.csv', header=None)
X = np.array(data_csv[[1,2]])
y = np.array(data_csv[0])

clf = SVC(C=1000000, kernel='linear',random_state=241)
clf.fit(X, y) 
