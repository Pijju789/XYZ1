# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 12:24:44 2018

@author: 7000320
"""
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col
from pyspark import SparkContext
sc = SparkContext()
from pyspark import spark

data = [(0, Vectors.dense([-1.0, -1.0 ]),),
        (1, Vectors.dense([-1.0, 1.0 ]),),
        (2, Vectors.dense([1.0, -1.0 ]),),
        (3, Vectors.dense([1.0, 1.0]),)]

df = spark.createDataFrame(data, ["id", "features"])

brp = BucketedRandomProjectionLSH(inputCol="features", outputCol="hashes",
                                  seed=12345, bucketLength=1.0)

spark = SparkSession.builder \
     .master("local") \
     .appName("Word Count") \
     .config("spark.some.config.option", "some-value") \
     .getOrCreate()
     
df2 = spark.createDataFrame([(2,), (5,), (5,)], ('age',))


##JAVA home needs to be reset

import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets,svm,metrics

digits = datasets.load_digits()
images_and_labels = list(zip(digits.images, digits.target))

for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)
    
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

classifier = svm.SVC(gamma= 0.001)
classifier.fit(data[:n_samples//2], digits.target[:n_samples//2])


##Prediction

expected = digits.target[n_samples // 2:]
predicted = classifier.predict(data[n_samples // 2:])

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))


images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()