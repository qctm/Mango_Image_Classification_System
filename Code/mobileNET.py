import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3, MobileNet, VGG16, DenseNet121, EfficientNetB7
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from keras import layers
from keras import models

path = "../newmodelA/img/"

categories = ['Ca','Chuoi','Dua','Tao']   

# add lable
data = []
labels = []
imagePaths = []

HEIGHT = 128 #
WIDTH = 128 #
N_CHANNELS = 3

for k, category in enumerate(categories):
    for f in os.listdir(path+category):
        imagePaths.append([path+category+'/'+f, k])

import random
random.shuffle(imagePaths)
print(imagePaths[:10])

for imagePath in imagePaths:
    image = cv2.imread(imagePath[0])
    image = cv2.resize(image, (WIDTH, HEIGHT))
    data.append(image)

    label = imagePath[1]
    labels.append(label)
    
# 1.6
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# Chia du lieu
from sklearn.model_selection import train_test_split
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=32)

#1.9
trainY = np_utils.to_categorical(trainY, len(categories))

print(trainX.shape)
print(testX.shape)
print(trainY.shape)
print(testY.shape)

# Kien truc MobileNet
EPOCHS = 30
INIT_LR = 1e-3
BS = 32

class_name = categories

print("[INFO] compiling model...")
mobileNet = MobileNet(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
for layer in mobileNet.layers:
  layer.trainable = False

model = Sequential()
model.add(mobileNet)
model.add(GlobalAveragePooling2D())
model.add(layers.Dropout(0.5))
model.add(layers.Flatten())
model.add(layers.Dense(len(class_name), activation='softmax'))

opt = tf.keras.optimizers.legacy.Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.summary()

model.fit(trainX, trainY, batch_size=BS, epochs=EPOCHS, verbose=1)

# 1.12 danh gia mo hinh
from numpy import argmax
from sklearn.metrics import confusion_matrix, accuracy_score

pred = model.predict(testX)
predictions = argmax(pred, axis=1)

cm = confusion_matrix(testY, predictions)

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title("model confusion matrix")
fig.colorbar(cax)
ax.set_xticklabels(['']+ categories)
ax.set_yticklabels(['']+ categories)

for i in range(len(categories)):
    for j in range(len(categories)):
        ax.text(i, j, cm[i, j], va='center', ha='center')

plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

accuracy = accuracy_score(testY, predictions)
print("Accuracy: %.2f%%" % (accuracy*100.0))

model.save("mKhac.h5")