# load imports
from os import path, makedirs
import numpy as np
import pickle
import time
from utils import convert_to_higher_id, convert_IDs_to_num, load_tiny_imgnet

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
output_dir = path.join('/scratch/users/ieisenbe/CBMM/output', 
                       'IMGNET_' + datetime)
makedirs(output_dir)
        
# load data
print("Loading Data")
(xtrain, classtrain, bbtrain), \
(xval, classval, bbval), \
(xtest, classtest, bbtest) = load_tiny_imgnet()


# create dictionary of models with their test data
models = {}
# create model architecture
print('Model setup')
image_shape = (64,64,3)
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
base_model=Model(input_img, drop2)

# for bounding box
readout_bb = Dense(4, activation='linear', name='readout')
bb_model=Model(base_model.input, readout_bb(base_model.layers[-1].output))
models['bb'] = {'model': bb_model,
                'test_data': (bbtrain, bbval, bbtest)}
# for classification at different hierarchies
hierarchical_models = {}
for level in range(3):
    # replace labels with higher-order classes
    hclass_train = convert_to_higher_id('wordnet.is_a.txt', classtrain, level)
    hclass_val = convert_to_higher_id('wordnet.is_a.txt', classval, level)
    hclass_test = convert_to_higher_id('wordnet.is_a.txt', classtest, level)
    # convert strings to numeric ids
    hclass_train, lookup = convert_IDs_to_num(hclass_train)
    hclass_val, lookup = convert_IDs_to_num(hclass_val)
    hclass_test, lookup = convert_IDs_to_num(hclass_test)
    # convert to one-hot representation
    num_classes = len(np.unique(hclass_train))
    onehot_train = keras.utils.to_categorical(hclass_train, num_classes)
    onehot_val = keras.utils.to_categorical(hclass_val, num_classes)
    onehot_test = keras.utils.to_categorical(hclass_test, num_classes)
    
    readout_class = Dense(num_classes, activation='softmax', name='readout')
    class_model=Model(base_model.input, 
                      readout_class(base_model.layers[-1].output))
    models['hierarchy_%s' % level] = {'model': class_model,
                                      'test_data': (onehot_train, 
                                                    onehot_val, 
                                                    onehot_test)}
    


print('Training the models')
# fit model
batch_size = 300
epochs = 100

# This will do preprocessing and realtime data augmentation:
datagen = ImageDataGenerator(
    zoom_range=.2,
    rotation_range=0,  
    width_shift_range=0.1, 
    height_shift_range=0.1,
    horizontal_flip=True) 


for name, val in models.items():
    model, (ytrain, yval, ytest) = val.values()
    
    # compile model
    opt = keras.optimizers.Adam(lr=0.001)
    if 'hierarchy' in name:
        model.compile(optimizer=opt, 
                          loss='categorical_crossentropy', 
                          metrics=['accuracy'])
    else:
        model.compile(optimizer=opt, 
                  loss='mean_squared_error', 
                  metrics=['accuracy'])
    
    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(xtrain, seed=202013)

    # run model
    makedirs(path.join(output_dir,'%s_checkpoints' % name)) # save checkpoint dir
    save_classcallback = ModelCheckpoint(path.join(output_dir,
                                              '%s_checkpoints' % name,
                                              '%s_weights.{epoch:03d}.h5' % name),
                                        save_weights_only=True, period=5)
        
    out = model.fit_generator(datagen.flow(xtrain, ytrain, batch_size=batch_size),
                              steps_per_epoch=xtrain.shape[0] // batch_size,
                              epochs=epochs,
                              validation_data=(xval, yval),
                              callbacks=[save_classcallback])

    model.save(path.join(output_dir, '%s.h5' % name))
    pickle.dump(out.history, open(path.join(output_dir, 
                                            '%s_modelhistory.pkl' % name), 
                                            'wb'))
    

