# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 15:15:18 2018

@author: 7000320
"""

import pandas as pd
import numpy as np
##%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import pygal
import plotly
from random import random , randint

Hacker_train1= pd.read_csv('C:/Users/7000320/Desktop/Hackerearth/criminal_train.csv',index_col=0)
Hacker_train1.head(5)
Hacker_train1.describe()

Hacker_test1= pd.read_csv('C:/Users/7000320/Desktop/Hackerearth/criminal_test.csv',index_col=0)
Hacker_test1.head(5)

#####Data Cleaning##


Hacker_train1.corr()
Hacker_train1.corr('spearman')

sns.countplot(x='Criminal', data=Hacker_train1);
pd.Categorical(Hacker_train1)

X = Hacker_train1.drop(['Criminal'], axis=1)
y = Hacker_train1["Criminal"]

#split into train and test
from sklearn.model_selection import train_test_split
X_Hacker_train1, X_Hacker_test1, y_Hacker_train1, y_Hacker_test1 = train_test_split(X,y,test_size = .33, random_state = 1)

# Feature Scaling###
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_Hacker_train1 = sc.fit_transform(X_Hacker_train1)
X_Hacker_test1 = sc.fit_transform(X_Hacker_test1)


final_Hacker_test1 = sc.transform(Hacker_test1) ## Necessary modules for creating models. 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score,classification_report, precision_recall_curve, confusion_matrix
from sklearn.metrics import confusion_matrix


##Logistic regression

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_Hacker_train1,y_Hacker_train1)
y_pred = logreg.predict(X_Hacker_test1)
logreg_accy = round(accuracy_score(y_pred,y_Hacker_test1), 3)
print (logreg_accy)

print (classification_report(y_Hacker_test1, y_pred, labels=logreg.classes_))
print (confusion_matrix(y_pred, y_Hacker_test1))

from sklearn.metrics import matthews_corrcoef
matthews_corrcoef(y_Hacker_test1, y_pred, sample_weight=None)
#####gives 0.42252971448921939 #####

from imblearn.over_sampling import SMOTE, ADASYN
X_Hacker_train1_resampled, y_Hacker_train1_resampled = SMOTE().fit_sample(X, y)
X_Hacker_test1_resampled, y_Hacker_test1_resampled = SMOTE().fit_sample(X, y)
import collections
from collections import Counter
print(sorted(Counter(y_Hacker_train1_resampled).items()))
print(sorted(Counter(y_Hacker_test1_resampled).items()))
[(0, 4674), (1, 4674), (2, 4674)]
from sklearn.svm import LinearSVC
clf_smote = LinearSVC().fit(X_Hacker_train1_resampled, y_Hacker_train1_resampled)
X_Hacker_train1_resampled, y_Hacker_train1_resampled = ADASYN().fit_sample(X, y)
print(sorted(Counter(y_Hacker_train1_resampled).items()))
[(0, 4673), (1, 4662), (2, 4674)]
clf_adasyn = LinearSVC().fit(X_Hacker_test1_resampled, y_Hacker_test1_resampled)

clf_adasyn


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_Hacker_train1_resampled,y_Hacker_train1_resampled)
y_pred1 = logreg.predict(X_Hacker_test1_resampled)
logreg_accy1 = round(accuracy_score(y_pred1,y_Hacker_test1_resampled), 3)
print (logreg_accy1)

from sklearn.metrics import matthews_corrcoef
matthews_corrcoef(y_Hacker_test1_resampled, y_pred1, sample_weight=None)



######RANDOM FOREST########
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from tabulate import tabulate
clf = RandomForestRegressor()
Model = clf.fit(X_Hacker_train1, y_Hacker_train1)


importance = clf.feature_importances_
importance = pd.X_Hacker_train1(importance, index=X_Hacker_train1.columns, 
                          columns=["Importance"])
headers = ["name", "score"]
values = sorted(zip(X_Hacker_train1, clf.feature_importances_), key=lambda x: x[1] * -1)
print(tabulate(values, headers, tablefmt="plain"))


features=Hacker_train1.iloc[:,1:70]
importances = clf.feature_importances_
indices = np.argsort(importances)

plt.figure(1)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')

rf = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=0)
rf.fit(X_Hacker_train1, y_Hacker_train1)
randomforest = RandomForestClassifier(n_estimators = 100, oob_score = True, random_state = 0)
randomforest.fit(X_Hacker_train1, y_Hacker_train1)

y_pred = randomforest.predict(X_Hacker_test1)
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, pearsonr

spearman = spearmanr(y_Hacker_test1, y_pred)
spearman

random_accy = round(accuracy_score(y_pred, y_Hacker_test1), 3)
print (random_accy)
from sklearn.metrics import matthews_corrcoef
matthews_corrcoef(y_Hacker_test1, y_pred, sample_weight=None)

randomforest = RandomForestClassifier(n_estimators = 500, oob_score = True, random_state = 1)
randomforest.fit(X_Hacker_train1_resampled, y_Hacker_train1_resampled)
y_pred2 = randomforest.predict(X_Hacker_test1_resampled)
spearman = spearmanr(y_Hacker_test1_resampled, y_pred2)
spearman
matthews_corrcoef(y_Hacker_test1_resampled, y_pred2, sample_weight=None)

random_accy1 = round(accuracy_score(y_pred2, y_Hacker_test1_resampled), 3)
print (random_accy)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_pred, y_Hacker_test1)

##Decision tree

#Decision Tree
from sklearn.tree import DecisionTreeClassifier

dectree = DecisionTreeClassifier( max_depth=5, 
                                class_weight = 'balanced',
                                min_weight_fraction_leaf = 0.01)
dectree.fit(X_Hacker_train1, y_Hacker_train1)
y_pred_dectree = dectree.predict(X_Hacker_test1)
dectree_accy = round(accuracy_score(y_pred_dectree, y_Hacker_test1), 3)
print(dectree_accy)

matthews_corrcoef(y_Hacker_test1, y_pred_dectree, sample_weight=None)

dectree.fit(X_Hacker_train1_resampled, y_Hacker_train1_resampled)
y_pred_dectree = dectree.predict(X_Hacker_test1_resampled)
dectree_accy1 = round(accuracy_score(y_pred_dectree, y_Hacker_test1_resampled), 3)
print(dectree_accy1)
matthews_corrcoef(y_Hacker_test1_resampled, y_pred_dectree, sample_weight=None)

############submission#####

test_prediction = randomforest.predict(Hacker_test1)

test.shape
test.head()
test.to_csv( 'sample_submission.csv' , index = False )