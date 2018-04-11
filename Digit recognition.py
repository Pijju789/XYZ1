# -*- coding: utf-8 -*-
"""
Created on Mon Apr 02 17:04:44 2018

@author: Pijush
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import scipy.misc
from scipy.ndimage import imread
##import tensorflow as tf
import keras

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

##train = pd.read_csv("c:/Users/7000320/Desktop/Backup IT011/Pijush Resources/DataSets/MNIST/train-images.csv")

from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images.shape
train_labels

from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))


##Compiler step

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])


##scaling and normalizing

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

## Categorical encoding of labels

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

##Network fit

network.fit(train_images, train_labels, epochs=2, batch_size=128)
test_loss, test_acc = network.evaluate(test_images, test_labels)

print('test_acc:', test_acc)


train=os.listdir("C:/Users/7000320/Desktop/Backup IT011/Pijush Resources/DataSets/MNIST")
filepath = "C:/Users/7000320/Desktop/Backup IT011/Pijush Resources/DataSets/MNIST"

import gzip
import pickle


with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f)
train_x, train_y = train_set   
import matplotlib.cm as cm
import matplotlib.pyplot as plt


plt.imshow(train_x[1].reshape((28, 28)), cmap=cm.Greys_r)
plt.show()
    
    
##images = []
##labels = []
##for i in train_images:
##    image = scipy.ndimage.imread(train_images)
##    images.append(image)
##    labels.append(0) #for cat images

    