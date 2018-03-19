# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 12:37:27 2018

@author: Pijush
"""

import pandas as pd
import numpy as np
import future
import np_utils
import theano
import keras
##import xgboost as xgb
from tqdm import tqdm
from sklearn.svm import SVC
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

Spooky_train = pd.read_csv("C:/Users/7000320/Desktop/Backup IT011/Pijush Resources/DataSets/Spooky author/train.csv")
Spooky_test = pd.read_csv("C:/Users/7000320/Desktop/Backup IT011/Pijush Resources/DataSets/Spooky author/test.csv")

def multiclass_logloss(actual, predicted, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    :param actual: Array containing the actual target classes
    :param predicted: Matrix with class predictions, one probability per class
    """
    # Convert 'actual' to a binary array if it's not already:
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota

##Encoding authors with level encoder##
    
lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(Spooky_train.author.values)

## Breaking into Train and validation set
xtrain, xvalid, ytrain, yvalid = train_test_split(Spooky_train.text.values, y, 
                                                  stratify=y, 
                                                  random_state=42, 
                                                  test_size=0.1, shuffle=True)

print (xtrain.shape)
print (xvalid.shape)

##tfidf factor##

tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')

tfv.fit(list(xtrain)+list(xvalid))
xtrain_tfv = tfv.transform(xtrain)
xvalid_tfv= tfv.transform(xvalid)
xtrain_tfv

##fitting logistic regression model##

clf = LogisticRegression(C=1.0)
clf.fit(xtrain_tfv, ytrain)
predictions = clf.predict_proba(xvalid_tfv)

print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))

