# -*- coding: utf-8 -*-
"""
@author: DarthVictor
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import cross_val_score
#X = np.array([[1, 2], [3, 4], [5, 6]])
#y = np.array([-3, 1, 10])
#clf = RandomForestRegressor(n_estimators=100)
#clf.fit(X, y)
#predictions = clf.predict(X)

data = pd.read_csv('abalone.csv')

data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

num_of_columns = len(data.columns.values)

X = data[range(0, num_of_columns - 1)]
y = data[data.columns.values[num_of_columns - 1]]

n_blocks = 5
kf = KFold(n_splits=n_blocks, random_state=1, shuffle=True)

n_estimators_max = 50
for n_estimators in range(1, n_estimators_max + 1):
    clf = RandomForestRegressor(random_state=1, n_estimators=n_estimators)
    score = cross_val_score(clf, X, y, cv=kf, scoring=make_scorer(r2_score)).mean()
    print(n_estimators, score)