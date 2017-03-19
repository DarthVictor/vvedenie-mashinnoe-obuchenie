# -*- coding: utf-8 -*-
"""
@author: DarthVictor
"""

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
close_prices_csv = pd.read_csv('close_prices.csv')
X_data = close_prices_csv[range(1,31)]
NumOfCOmponents = 10

pca = PCA(n_components = NumOfCOmponents)
pca.fit(X_data)
print(pca.explained_variance_ratio_) 
var_ratio = pca.explained_variance_ratio_


total_explaned_variance = 0
VarianceToExplain = 0.9

for i in range(0, NumOfCOmponents):
    total_explaned_variance += var_ratio[i]
    if total_explaned_variance >= VarianceToExplain :
        print(i + 1)
        break
X_predicted = pca.transform(X_data)

djia_index = pd.read_csv('djia_index.csv')['^DJI']

print(round(np.corrcoef(X_predicted[:,0], djia_index)[1,0], 2))

first_component = np.array(pca.components_[0])
max_equity_col = first_component.argmax()
print(close_prices_csv[[max_equity_col+1]].axes[1][0])