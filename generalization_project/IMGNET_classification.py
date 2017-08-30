# load imports
from os import path, makedirs
import numpy as np
import pickle
import time
from utils import convert_IDs_to_num, load_tiny_imgnet

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

# convert string ids to numeric ids
classtrain, lookup = convert_IDs_to_num(classtrain)
classval, lookup = convert_IDs_to_num(classval)

# reshape data
num_classes = len(np.unique(classtrain))
classtrain = keras.utils.to_categorical(classtrain, num_classes)
classval = keras.utils.to_categorical(classval, num_classes)


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
# for classification
readout_class = Dense(num_classes, activation='softmax', name='readout')(drop2)
# for bounding box
readout_bb = Dense(4, activation='linear', name='readout')(drop2)

class_model=Model(input_img, readout_class)
bb_model=Model(input_img, readout_bb)

opt = keras.optimizers.Adam(lr=0.001)
class_model.compile(optimizer=opt, 
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'])
bb_model.compile(optimizer=opt, 
                  loss='mean_squared_error', 
                  metrics=['accuracy'])


# fit model
batch_size = 300
epochs = 200

print('Training the model')
start = time.time()
makedirs(path.join(output_dir,'classification_checkpoints')) # save checkpoint dir
save_classcallback = ModelCheckpoint(path.join(output_dir,
                                          'classification_checkpoints',
                                          'classification_weights.{epoch:03d}.h5'),
                                    save_weights_only=True, period=5)
save_bbcallback = ModelCheckpoint(path.join(output_dir,
                                          'bb_checkpoints',
                                          'bb_weights.{epoch:03d}.h5'),
                                    save_weights_only=True, period=5)
# This will do preprocessing and realtime data augmentation:
datagen = ImageDataGenerator(
    zoom_range=.2,
    rotation_range=0,  
    width_shift_range=0.1, 
    height_shift_range=0.1,
    horizontal_flip=True) 

# Compute quantities required for feature-wise normalization
# (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(xtrain, seed=202013)

class_out = class_model.fit_generator(datagen.flow(xtrain, classtrain,
                                                   batch_size=batch_size),
                                    steps_per_epoch=xtrain.shape[0] // batch_size,
                                    epochs=epochs,
                                    validation_data=(xval, classval),
                                    callbacks=[save_classcallback])

class_model.save(path.join(output_dir, 'classification_model.h5'))
pickle.dump(class_out.history, open(path.join(output_dir, 
                                        'classification_modelhistory.pkl'), 
                                        'wb'))

datagen.fit(xtrain, seed=202013)
bb_out = bb_model.fit_generator(datagen.flow(xtrain, bbtrain,
                                             batch_size=batch_size),
                                steps_per_epoch=xtrain.shape[0] // batch_size,
                                epochs=epochs,
                                validation_data=(xval, bbval),
                                callbacks=[save_bbcallback])

bb_model.save(path.join(output_dir, 'bb_model.h5'))
pickle.dump(bb_out.history, open(path.join(output_dir, 
                                        'bb_modelhistory.pkl'), 
                                        'wb'))

end = time.time()
print("Model took %0.2f seconds to train"%(end - start))


