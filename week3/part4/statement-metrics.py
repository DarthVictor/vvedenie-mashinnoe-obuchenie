# -*- coding: utf-8 -*-
"""
@author: DarthVictor
"""

import numpy as np
import pandas as pd

#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#
#classification_csv = pd.read_csv('classification.csv')
#
#actual_classes = classification_csv['true']
#predicted_classes = classification_csv['pred']
#
#TP = 0
#FP = 0
#FN = 0
#TN = 0
#
#for i in xrange(len(actual_classes)):
#    if predicted_classes[i] == 1 and actual_classes[i] == 1:
#        TP += 1
#    elif predicted_classes[i] == 1 and actual_classes[i] == 0:
#        FP += 1
#    elif predicted_classes[i] == 0 and actual_classes[i] == 1:
#        FN += 1
#    elif predicted_classes[i] == 0 and actual_classes[i] == 0:
#        TN += 1
#        
#print(TP, FP, FN, TN)
#
##Accuracy (доля верно угаданных) — sklearn.metrics.accuracy_score
##Precision (точность) — sklearn.metrics.precision_score
##Recall (полнота) — sklearn.metrics.recall_score
##F-мера — sklearn.metrics.f1_score
#print(
#    round(accuracy_score(actual_classes, predicted_classes), 2),
#    round(precision_score(actual_classes, predicted_classes), 2),
#    round(recall_score(actual_classes, predicted_classes), 2),
#    round(f1_score(actual_classes, predicted_classes), 2)
#)


from sklearn.metrics import roc_auc_score, precision_recall_curve
scores_csv = pd.read_csv('scores.csv')
#для логистической регрессии — вероятность положительного класса (колонка score_logreg),
#для SVM — отступ от разделяющей поверхности (колонка score_svm),
#для метрического алгоритма — взвешенная сумма классов соседей (колонка score_knn),
#для решающего дерева — доля положительных объектов в листе (колонка score_tree).

actual = scores_csv['true']
score_logreg = scores_csv['score_logreg']
score_svm = scores_csv['score_svm']
score_knn = scores_csv['score_knn']
score_tree = scores_csv['score_tree']

#print(
#    roc_auc_score(actual, score_logreg),
#    roc_auc_score(actual, score_svm),
#    roc_auc_score(actual, score_knn),
#    roc_auc_score(actual, score_tree)        
#)
# score_logreg
def getMaxPrecisionForRecall(y_true, y_score, min_recall=0.7):
    curve = precision_recall_curve(y_true, y_score)
    max_precision = 0
    for i in xrange(len(curve[0])):
        if max_precision < curve[0][i] and curve[1][i] >= min_recall:
            max_precision = curve[0][i]
    return max_precision
        
print(
    'score_logreg', getMaxPrecisionForRecall(actual, score_logreg),
    'score_svm', getMaxPrecisionForRecall(actual, score_svm),
    'score_knn', getMaxPrecisionForRecall(actual, score_knn),
    'score_tree', getMaxPrecisionForRecall(actual, score_tree)        
)
#score_tree
