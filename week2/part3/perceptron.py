# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 12:45:42 2017

@author: DarthVictor
"""
import pandas
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data_train = pandas.read_csv('perceptron-train.csv', header=None)
X_train = np.array(data_train[[1,2]])
y_train = np.array(data_train[0])

data_test = pandas.read_csv('perceptron-test.csv', header=None)
X_test = np.array(data_test[[1,2]])
y_test = np.array(data_test[0])

clf = Perceptron(random_state=241)
clf.fit(X_train, y_train)
accuracy_unscaled = accuracy_score(clf.predict(X_test), y_test)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf_scaled = Perceptron(random_state=241)
clf_scaled.fit(X_train_scaled, y_train)
accuracy_scaled = accuracy_score(clf_scaled.predict(X_test_scaled), y_test)

print(accuracy_scaled - accuracy_unscaled)