# -*- coding: utf-8 -*-

import pandas
import numpy as np
import re
import collections
from sklearn.tree import DecisionTreeClassifier

data_dirty = pandas.read_csv('titanic.csv', index_col='PassengerId')
print('data >>>>>> ')
#print(len(data)) #891
#print(len(data[np.isfinite(data['Pclass'])])) #891
#print(len(data[np.isfinite(data['Fare'])])) #891
#print(len(data[data['Sex'] == 'male']) + len(data[data['Sex'] == 'female'])) #891

#print(len(data[np.isfinite(data['Age'])])) #714

df = data_dirty[np.isfinite(data_dirty['Age'])]

data = pandas.DataFrame.from_dict({
    'Pclass': df['Pclass'], 
    'Fare': df['Fare'], 
    'Age': df['Age'], 
    'Sex': df['Sex'].map(lambda s: 1 if s=='male' else 0 )}
)
target = df['Survived']

X = np.array([data['Pclass'], data['Fare'], data['Age'], data['Sex']]).transpose()
y = np.array(target)
clf = DecisionTreeClassifier(random_state=241)
clf.fit(X, y)
print(clf.feature_importances_)