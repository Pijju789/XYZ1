# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 11:57:59 2018

@author: Pijush
"""
from keras.layers import Embedding
embedding_layer= Embedding(1000,64)

##Importing IMDB Datasets

from keras.datasets import imdb
from keras import preprocessing

# Number of words to consider as features
max_features = 1000
# Cut texts after this number of words 
# (among top max_features most common words)
maxlen = 20

# Load the data as lists of integers.
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# This turns our lists of integers
# into a 2D integer tensor of shape `(samples, maxlen)`
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

##Sequencial data building

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
# We specify the maximum input length to our Embedding layer
# so we can later flatten the embedded inputs
model.add(Embedding(1000, 8, input_length=maxlen))
# After the Embedding layer, 
# our activations have shape `(samples, maxlen, 8)`.

# We flatten the 3D tensor of embeddings 
# into a 2D tensor of shape `(samples, maxlen * 8)`
model.add(Flatten())

# We add the classifier on top
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()

history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2)




## IF word vectors are less you can impute/add external glove directory
import numpy as np
import os
glove_dir = 'C:/Users/7000320/Desktop/Backup IT011/Pijush Resources/DataSets/word2vec'

embeddings_index = {}
f = open(os.path.join(glove_dir, 'crawl-300d-2M.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

