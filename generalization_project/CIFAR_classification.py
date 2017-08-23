# load imports
from glob import glob
from os import path, makedirs
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import numpy as np
import pickle
import time

import keras
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dropout, Flatten
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# setup
datetime = time.strftime("%d-%m-%Y-%H:%M:%S")
output_dir = path.join('output', datetime)
makedirs(output_dir)

# helper functions
def get_conv_layer(filters, layer_input):
    layer = Conv2D(input_shape=layer_input.shape.as_list(),
                   filters=filters,
                   kernel_size=3,
                   padding='same',
                   activation='relu')(layer_input)
    layer = Conv2D(filters=filters,
                   kernel_size=3,
                   padding='same',
                   activation='relu')(layer)
    layer = MaxPooling2D()(layer)
    layer = Dropout(.2)(layer)
    return layer

def split_val(x_train, y_train):
    split = len(x_train)*4//5
    (x_train, y_train), (x_val, y_val) = (x_train[:split], y_train[:split]), \
                                         (x_test[:split], y_test[:split])
    return (x_train, y_train), (x_val, y_val)                    
    
def plot_cifar10(X,Y):
    num_classes = len(Y[0,:])
    # plot random images from each class
    class_names = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']
    fig = plt.figure(figsize=(8,3))
    for i in range(num_classes):
        ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
        idx = np.where(X[:]==i)[0]
        features_idx = Y[idx,::]
        img_num = np.random.randint(features_idx.shape[0])
        im = features_idx[img_num,::]
        ax.set_title(class_names[i])
        plt.imshow(im)
    plt.show()

def save_weights(model, filey):
    weights = [l.get_weights() for l in model.layers]
    pickle.dump(weights, open(filey,'wb'))
    
def load_weights(model, filey):
    weights = pickle.load(open(filey,'rb'))
    for i,l in enumerate(model.layers):
        l.set_weights(weights[i])
        
    
# load data
from keras.datasets import cifar10, cifar100
datasets = {'cifar10': cifar10.load_data,
            'cifar100_fine': lambda: cifar100.load_data('fine'),
            'cifar100_coarse': lambda: cifar100.load_data('coarse')}
for data, load_fun in datasets.items():
    print('*'*100)
    print('Loading %s' % data)
    print('*'*100)
    (x_train, y_train), (x_test, y_test) = load_fun()
    (x_train, y_train), (x_val, y_val) = split_val(x_train, y_train)
    
    # reshape data
    num_classes = len(np.unique(y_train))
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    # set up model
    print('Model setup')
    image_shape = (32,32,3)
    input_img = Input(shape=image_shape)
    layer1 = get_conv_layer(32, input_img)
    layer2 = get_conv_layer(64, layer1)
    layer3 = get_conv_layer(128, layer2)
    flatten = Flatten()(layer3)
    full1 = Dense(512, activation='relu')(flatten)
    drop1 = Dropout(.5)(full1)
    full2 = Dense(256, activation='relu')(drop1)
    drop2 = Dropout(.5)(full2)
    full3 = Dense(num_classes, activation='softmax')(drop2)
    
    opt = keras.optimizers.Adam(lr=0.001)
    
    model=Model(input_img, full3)
    model.compile(optimizer=opt, 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
    
    # load/save initial weights
    try:
        print('Loading previous weight initialization')
        load_weights(model, path.join(output_dir,'initial_weights.pkl'))
    except IOError:
        print('Saving weight initalization')
        save_weights(model, path.join(output_dir,'initial_weights.pkl'))
    
    # fit model
    batch_size = 128
    epochs = 300
    data_augmentation = True
    
    print('Training the model')
    start = time.time()
    save_callback = ModelCheckpoint(path.join(output_dir,
                                              '%s_weights.{epoch:02d}' % data),
                                    save_weights_only=True, period=1)

    if not data_augmentation:
        print('Not using data augmentation.')
        out = model.fit(x_train, y_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=(x_val, y_val),
                          shuffle=True,
                          callbacks=[save_callback])
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            zoom_range=.2,
            rotation_range=0,  
            width_shift_range=0.1, 
            height_shift_range=0.1,
            horizontal_flip=True) 
    
        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)
        
        out = model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        callbacks=[save_callback])
    end = time.time()
    print("Model took %0.2f seconds to train"%(end - start))
    model.save(path.join(output_dir, '%s_model.h5' % data))
    pickle.dump(out, open(path.join(output_dir, '%s_modelinfo.pkl' % data)))