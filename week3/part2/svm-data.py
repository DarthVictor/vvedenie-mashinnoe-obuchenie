# -*- coding: utf-8 -*-
"""
@author: DarthVictor
"""

import numpy as np
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.model_selection import KFold, GridSearchCV

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
newsgroups = datasets.fetch_20newsgroups(
                    subset='all', 
                    categories=['alt.atheism', 'sci.space']
             )

#
#X = newsgroups.data
#y = newsgroups.target
#
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(newsgroups.data)
y = newsgroups.target
feature_mapping = vectorizer.get_feature_names()
#print (feature_mapping)
#
#
#grid = {'C': np.power(10.0, np.arange(-5, 6))}
#cv = KFold(n_splits=5, shuffle=True, random_state=241)
#clf = SVC(kernel='linear', random_state=241)
#gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
#gs.fit(X, y)
#
#print(gs.grid_scores_)

#grid_scores = [
#mean: 0.55263, std: 0.02812, params: {'C': 1.0000000000000001e-05}, 
#mean: 0.55263, std: 0.02812, params: {'C': 0.0001},
#mean: 0.55263, std: 0.02812, params: {'C': 0.001},
#mean: 0.55263, std: 0.02812, params: {'C': 0.01},
#mean: 0.95017, std: 0.00822, params: {'C': 0.10000000000000001},
#mean: 0.99328, std: 0.00455, params: {'C': 1.0}, 
#mean: 0.99328, std: 0.00455, params: {'C': 10.0},
#mean: 0.99328, std: 0.00455, params: {'C': 100.0},
#mean: 0.99328, std: 0.00455, params: {'C': 1000.0},
#mean: 0.99328, std: 0.00455, params: {'C': 10000.0},
#mean: 0.99328, std: 0.00455, params: {'C': 100000.0}
#]

#C = 1.0

clf_optimal = SVC(C=1.0, kernel='linear', random_state=241)
clf_optimal.fit(X, y)
ind = np.argsort(np.absolute(np.asarray(clf_optimal.coef_.todense())).reshape(-1))[-10:]
words = [vectorizer.get_feature_names()[i] for i in ind]
print(" ".join(sorted(words)))