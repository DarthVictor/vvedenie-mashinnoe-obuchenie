# -*- coding: utf-8 -*-
"""
@author: DarthVictor
"""

import numpy as np
import math

#clf.fit(X, y)
#predictions = clf.predict(X)
import pandas as pd
data = pd.read_csv('gbm-data.csv')
X = data.values[:,1:]
y = data.values[:,0]

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)

from sklearn.metrics import log_loss
#from sklearn.ensemble import GradientBoostingClassifier
#
#def sigmoid(y_pred):
#    res = []
#    for y_pred_i in y_pred:
#        res.append(1.0 / (1.0 + math.exp(-y_pred_i)))
#    return res
#
#learning_rates = [1, 0.5, 0.3, 0.2, 0.1] 
#result = {}
#for learning_rate in learning_rates:
#    gbc = GradientBoostingClassifier(
#            n_estimators=250, 
#            learning_rate=learning_rate,
#            verbose=True,
#            random_state=241)
#
#    gbc.fit(X_train, y_train)
#    
#    staged_decision_train = gbc.staged_decision_function(X_train)   
#    train_result = []
#    for t in staged_decision_train:
#        train_result.append(log_loss(y_train, sigmoid(t)))
#    
#    staged_decision_test = gbc.staged_decision_function(X_test)
#    test_result = []
#    for t in staged_decision_test:
#        test_result.append(log_loss(y_test, sigmoid(t)))
#    print(learning_rate)
#    print(min(train_result), train_result.index(min(train_result)))
#    print(min(test_result), test_result.index(min(test_result)))
#    result[learning_rate] = [train_result, test_result]
#        
#    
# overfitting
# 0.53145079631906378, 36

from sklearn.ensemble import RandomForestClassifier

rfl = RandomForestClassifier(n_estimators=36, random_state=241)
rfl.fit(X_train, y_train)
y_pred = rfl.predict_proba(X_test)[:, 1]
test_loss = log_loss(y_test, y_pred)
print(test_loss)