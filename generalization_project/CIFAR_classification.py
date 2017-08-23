#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 14:38:11 2017

@author: ian
"""
from glob import glob
from os import path
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import numpy as np
import pickle
import time

import keras
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dropout, Flatten
from keras.layers import Dense, convolutional, pooling
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


# load data
from keras.datasets import cifar10
data_loc = path.join('Data/cifar10.pkl')
try:
    (x_train, y_train), (x_test, y_test) = pickle.load(open(data_loc,'rb'))
except IOError:    
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    pickle.dump([(x_train, y_train), (x_test, y_test)], open(data_loc,'wb'))
# reshape data
num_classes = len(np.unique(y_train))

# plot random images from each class
class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']
fig = plt.figure(figsize=(8,3))
for i in range(num_classes):
    ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
    idx = np.where(y_train[:]==i)[0]
    features_idx = x_train[idx,::]
    img_num = np.random.randint(features_idx.shape[0])
    im = features_idx[img_num,::]
    ax.set_title(class_names[i])
    plt.imshow(im)
plt.show()

# helper functions
def get_conv_layer(filters, layer_input, input_shape=None):
    layer = convolutional.Conv2D(input_shape=image_shape,
                             filters=filters,
                             kernel_size=3,
                             padding='same',
                             activation='relu')(layer_input)
    layer = pooling.MaxPool2D()(layer)
    layer = Dropout(.2)(layer)
    return layer

# set up model
image_shape = (32,32,3)
input_img = Input(shape=image_shape)
layer1 = get_conv_layer(32, input_img, image_shape)
layer2 = get_conv_layer(64, layer1)
flatten = Flatten()(layer2)
full1 = Dense(512, activation='relu')(flatten)
drop1 = Dropout(.5)(full1)
full2 = Dense(num_classes, activation='softmax')(drop1)

model=Model(input_img, layer1)
model.compile(optimizer='adam', 
          loss='categorical_crossentropy', 
          metrics=['accuracy'])

# fit model
batch_size = 32
epochs = 300
data_augmentation = True
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

start = time.time()
if not data_augmentation:
    print('Not using data augmentation.')
    out = model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      validation_data=(x_test, y_test),
                      shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        zoom_range=.2,
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True) 

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)
    
    out = model.fit_generator(datagen.flow(x_train, y_train,
                                 batch_size=batch_size),
                    steps_per_epoch=x_train.shape[0] // batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test))
end = time.time()
print("Model took %0.2f seconds to train"%(end - start))
model.save(path.join('output', 'cifar10_model.h5'))
pickle.dump(out, open(path.join('output', 'cifar10_modelinfo.pkl')))