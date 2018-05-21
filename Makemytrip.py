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

Train_data = pd.read_csv('C:/Users/pijus_000/Desktop/python/dataset/train.csv')
Test_data = pd.read_csv('C:/Users/pijus_000/Desktop/python/dataset/test.csv')

len(Train_data)
len(Test_data)

Train_data.describe()
Test_data.describe()

Train_data[Train_data.isnull().any(axis=1)]
Train_data.fillna(0)
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

Target = Train_data["P"]
X = Train_data.drop('P',1)
X= X.dropna()
Target = Target.dropna()
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
print (logreg_accy)


