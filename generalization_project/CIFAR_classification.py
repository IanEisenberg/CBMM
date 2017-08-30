# load imports
from os import path, makedirs
import numpy as np
import pickle
import time
from utils import arbitrary_load, split_val

import keras
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dropout, Flatten, BatchNormalization
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator

# helper functions
def get_conv_layer(filters, layer_input):
    layer = Conv2D(input_shape=layer_input.shape.as_list(),
                   filters=filters,
                   kernel_size=3,
                   padding='same',
                   activation='relu')(layer_input)
    layer = BatchNormalization(axis=3)(layer)
    layer = Conv2D(filters=filters,
                   kernel_size=3,
                   padding='same',
                   activation='relu')(layer)
    layer = BatchNormalization(axis=3)(layer)
    layer = MaxPooling2D()(layer)
    layer = Dropout(.25)(layer)
    return layer


# setup
datetime = time.strftime("%d-%m-%Y_%H-%M-%S")
# local directory
#output_dir = path.join('output', datetime)
# for sherlock
output_dir = path.join('/scratch/users/ieisenbe/CBMM/output', datetime)
makedirs(output_dir)
        
# load data
from keras.datasets import cifar10, cifar100
datasets = {'cifar10': cifar10.load_data,
            'cifar100_fine': lambda: cifar100.load_data('fine'),
            'cifar100_coarse': lambda: cifar100.load_data('coarse'),
            'cifar100_arbitrary': arbitrary_load}

for data, load_fun in datasets.items():
    print('*'*100)
    print('Loading %s' % data)
    print('*'*100)
    (x_train, y_train), (x_test, y_test) = load_fun()
    (x_train, y_train), (x_val, y_val) = split_val(x_train, y_train)
    
    # reshape data
    num_classes = len(np.unique(y_train))
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)

    
    # load model and weights or create model architecture
    try:
        print('Loading model architecture')
        base_model = load_model(path.join(output_dir, 'basemodel_architecture.h5'))
        # change readout layer
        base_out = base_model.layers[-2].output
        readout = Dense(num_classes, activation='softmax', name='readout')
        model = Model(inputs=base_model.input, outputs=readout(base_out))

    except IOError:
        # set up model
        print('Model setup')
        image_shape = (32,32,3)
        input_img = Input(shape=image_shape)
        layer1 = get_conv_layer(32, input_img)
        layer2 = get_conv_layer(64, layer1)
        layer3 = get_conv_layer(128, layer2)
        flatten = Flatten()(layer3)
        full1 = Dense(512, activation='relu')(flatten)
        batch1 = BatchNormalization()(full1)
        drop1 = Dropout(.5)(batch1)
        full2 = Dense(256, activation='relu')(drop1)
        batch2 = BatchNormalization()(full2)
        drop2 = Dropout(.5)(batch2)
        readout = Dense(num_classes, activation='softmax', name='readout')(drop2)
        model=Model(input_img, readout)
        print('Saving base model architecture')
        model.save(path.join(output_dir, 'basemodel_architecture.h5'))
        
    opt = keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=opt, 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
    
    # fit model
    batch_size = 128
    epochs = 200
    
    print('Training the model')
    start = time.time()
    makedirs(path.join(output_dir,'%s_checkpoints' % data)) # save checkpoint dir
    save_callback = ModelCheckpoint(path.join(output_dir,
                                              '%s_checkpoints' % data,
                                              '%s_weights.{epoch:03d}.h5' % data),
                                    save_weights_only=True, period=5)
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        zoom_range=.2,
        rotation_range=0,  
        width_shift_range=0.1, 
        height_shift_range=0.1,
        horizontal_flip=True,
        seed = 202013) 

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)
    
    out = model.fit_generator(datagen.flow(x_train, y_train,
                                 batch_size=batch_size),
                    steps_per_epoch=x_train.shape[0] // batch_size,
                    epochs=epochs,
                    validation_data=(x_val, y_val),
                    callbacks=[save_callback])
    end = time.time()
    print("Model took %0.2f seconds to train"%(end - start))
    model.save(path.join(output_dir, '%s_model.h5' % data))
    pickle.dump(out.history, open(path.join(output_dir, '%s_modelhistory.pkl'\
                                            % data), 'wb'))
