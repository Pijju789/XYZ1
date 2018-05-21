# -*- coding: utf-8 -*-
"""
Created on Sat May 19 11:37:18 2018
@author: pijus_000
"""

import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

Train_data = pd.read_csv('C:/Users/7000320/Desktop/Backup IT011/Pijush Resources/DataSets/Makemytrip/dataset/train.csv')
Test_data = pd.read_csv('C:/Users/7000320/Desktop/Backup IT011/Pijush Resources/DataSets/Makemytrip/dataset/test.csv')
sample_data = pd.read_csv('C:/Users/7000320/Desktop/Backup IT011/Pijush Resources/DataSets/Makemytrip/dataset/sample_submission.csv')
len(Train_data)
len(Test_data)

Train_data.describe()
Test_data.describe()

Train_data[Train_data.isnull().any(axis=1)]
Train_data = Train_data.fillna(0)

Test_data[Test_data.isnull().any(axis=1)]
Test_data = Test_data.fillna(0)
##Train_data.dropna()
##Visualization

sns.countplot(Train_data['B'])
Train_data[Train_data['B'] < 20 ]['B'].value_counts().sort_index().plot.line()
sns.countplot(Train_data['C'])
sns.countplot(Train_data['H'])
sns.countplot(Train_data['K'])
sns.countplot(Train_data['N'])
sns.countplot(Train_data['O'])


sns.distplot(Train_data['H'], bins=10, kde=False)
sns.distplot(Train_data['B'], bins=2, kde=False)
sns.distplot(Train_data['C'], bins=10, kde=False)
sns.distplot(Train_data['K'], bins=10, kde=False)
sns.distplot(Train_data['N'], bins=10, kde=False)
sns.distplot(Train_data['O'], bins=10, kde=False)

##Model Building
##pd.get_dummies(obj_df, columns=["body_style", "drive_wheels"], prefix=["body", "drive"]).head()
## Train Data encoding
from sklearn import preprocessing
Train_data["A"] = Train_data["A"].astype('category')
Train_data["A"] = Train_data["A"].cat.codes

Train_data["D"] = Train_data["D"].astype('category')
Train_data["D"] = Train_data["D"].cat.codes

Train_data["E"] = Train_data["E"].astype('category')
Train_data["E"] = Train_data["E"].cat.codes

Train_data["F"] = Train_data["F"].astype('category')
Train_data["F"] = Train_data["F"].cat.codes

Train_data["G"] = Train_data["G"].astype('category')
Train_data["G"] = Train_data["G"].cat.codes

Train_data["I"] = Train_data["I"].astype('category')
Train_data["I"] = Train_data["I"].cat.codes

Train_data["J"] = Train_data["J"].astype('category')
Train_data["J"] = Train_data["J"].cat.codes

Train_data["L"] = Train_data["L"].astype('category')
Train_data["L"] = Train_data["L"].cat.codes

Train_data["M"] = Train_data["M"].astype('category')
Train_data["M"] = Train_data["M"].cat.codes

## Test Data Encoding ##
Test_data["A"] = Test_data["A"].astype('category')
Test_data["A"] = Test_data["A"].cat.codes

Test_data["D"] = Test_data["D"].astype('category')
Test_data["D"] = Test_data["D"].cat.codes

Test_data["E"] = Test_data["E"].astype('category')
Test_data["E"] = Test_data["E"].cat.codes

Test_data["F"] = Test_data["F"].astype('category')
Test_data["F"] = Test_data["F"].cat.codes

Test_data["G"] = Test_data["G"].astype('category')
Test_data["G"] = Test_data["G"].cat.codes

Test_data["I"] = Test_data["I"].astype('category')
Test_data["I"] = Test_data["I"].cat.codes

Test_data["J"] = Test_data["J"].astype('category')
Test_data["J"] = Test_data["J"].cat.codes

Test_data["L"] = Test_data["L"].astype('category')
Test_data["L"] = Test_data["L"].cat.codes

Test_data["M"] = Test_data["M"].astype('category')
Test_data["M"] = Test_data["M"].cat.codes

Target = Train_data["P"]
X = Train_data.drop('P',1)

##Test Train split
X[X.isnull().any(axis=1)]
X = X.fillna(0)

from sklearn.cross_validation import train_test_split


X_train, X_test, Target_train, Target_test = train_test_split(
 X, Target, test_size=0.33, random_state=42)

## Logistic Regression##

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report, precision_recall_curve, confusion_matrix
logreg = LogisticRegression()
logreg.fit(X_train,Target_train)
Target_pred = logreg.predict(X_test)
logreg_accy = round(accuracy_score(Target_pred,Target_test), 3)
print (logreg_accy) ## 79.8% Accuracy

##cross validation

from sklearn import metrics, cross_validation
predicted_cv = cross_validation.cross_val_predict(logreg, X, Target, cv=10)
metrics.accuracy_score(Target, predicted_cv) ## 86.05 %

from sklearn.cross_validation import cross_val_score
accuracy_cv = cross_val_score(logreg, X, Target, cv=10,scoring='accuracy')
print (accuracy_cv)
print (cross_val_score(logreg, X, Target, cv=10,scoring='accuracy').mean())

from nltk import ConfusionMatrix 
print (ConfusionMatrix(list(Target), list(predicted_cv)))


## Decision tree ##
##Cross Validation


from sklearn.tree import DecisionTreeClassifier

dectree = DecisionTreeClassifier( max_depth=3, 
                                class_weight = 'balanced',
                                min_weight_fraction_leaf = 0.01)

depth = []
for i in range(3,20):
    clf = DecisionTreeClassifier(max_depth=i)
    # Perform 7-fold cross validation 
    scores = cross_val_score(estimator=clf, X=X, y=Target, cv=7, n_jobs=4)
    depth.append((i,scores.mean()))
print(depth)  ## max_depth 3 gives max score.

dectree.fit(X_train,Target_train)
dectree_pred = dectree.predict(X_test)
dectree_accy = round(accuracy_score(dectree_pred, Target_test), 3)
print(dectree_accy) ## 78.7 % Accuracy

## Random Forest ##

##Random Forest Cross-Validation##
##from sklearn.ensemble import RandomForest

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

rf1 = RandomForestClassifier()
print np.mean(cross_val_score(rf1, X, Target, cv=10))

param_grid = {
                 'n_estimators': [100, 200, 300, 500],
                 'max_depth': [2, 5, 7, 9]
             }

from sklearn.grid_search import GridSearchCV

grid_clf = GridSearchCV(rf1, param_grid, cv=10)
grid_clf.fit(X, Target)

rf = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=0)
rf.fit(X_train, Target_train)
rf.pred = rf.predict(X_test)
rf_accy = round(accuracy_score(rf.pred, Target_test), 3)

randomforest = RandomForestClassifier(n_estimators = 100, oob_score = True, random_state = 0)
randomforest.fit(X_train, Target_train)
randomforest_pred = randomforest.predict(X_test)
randomforest_accy = round(accuracy_score(randomforest_pred,Target_test), 3)
print(randomforest_accy) ## 82.5 Accuracy

from sklearn.metrics import matthews_corrcoef, spearman_corrcoef
#spearman = spearmanr(y_Hacker_test1_resampled, y_pred2)


##SVM##

from sklearn import svm
svm1 = svm.SVC(kernel='linear', C=1).fit(X_train, Target_train)
svm_pred = svm1.score(X_test,Target_test)
Target_pred1 = logreg.predict(Test_data) ## Prediction with Logistic regression
Target_pred2 = randomforest.predict(Test_data) ## Prediction with Random forest

##Submission##

Test_data['P'] = Target_pred2
Submission = Test_data[['id','P']]

Submission.to_csv( 'sample_submission.csv' , index = False )
