# -*- coding: utf-8 -*-
"""
@author: DarthVictor
"""

import numpy as np

import pandas as pd
train = pd.read_csv('salary-train.csv')
train['FullDescription'] = train['FullDescription'].str.lower().replace('[^a-zA-Z0-9]', ' ', regex = True)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=5)
X_train_words = vectorizer.fit_transform(train['FullDescription'])
X_test_words = vectorizer.transform(train['FullDescription'])

train['LocationNormalized'].fillna('nan', inplace=True)
train['ContractTime'].fillna('nan', inplace=True)

from sklearn.feature_extraction import DictVectorizer
enc = DictVectorizer()
X_train_categ = enc.fit_transform(train[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(train[['LocationNormalized', 'ContractTime']].to_dict('records'))

from scipy.sparse import hstack
X = hstack([
        X_test_words, 
        X_test_categ, 
])

from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0, random_state=241)
ridge.fit(X, train['SalaryNormalized'])

test = pd.read_csv('salary-test-mini.csv')
