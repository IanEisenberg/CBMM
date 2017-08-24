# load imports
from glob import glob
import h5py
from os import path, makedirs
import pandas as pd
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from utils.py import get_avg_rep, split_val

import keras
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dropout, Flatten, BatchNormalization
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


# labels
cifar10_labels = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']

cifar100_fine_labels = \
"""
apples, mushrooms, oranges, pears, sweet peppers, 
aquarium fish, flatfish, ray, shark, trout, 
beaver, dolphin, otter, seal, whale, 
orchids, poppies, roses, sunflowers, tulips, 
bottles, bowls, cans, cups, plates, 
clock, computer keyboard, lamp, telephone, television, 
bed, chair, couch, table, wardrobe, 
bee, beetle, butterfly, caterpillar, cockroach, 
bear, leopard, lion, tiger, wolf, 
bridge, castle, house, road, skyscraper, 
cloud, forest, mountain, plain, sea, 
camel, cattle, chimpanzee, elephant, kangaroo, 
fox, porcupine, possum, raccoon, skunk, 
crab, lobster, snail, spider, worm, 
baby, boy, girl, man, woman, 
crocodile, dinosaur, lizard, snake, turtle, 
hamster, mouse, rabbit, shrew, squirrel, 
maple, oak, palm, pine, willow, 
bicycle, bus, motorcycle, pickup truck, train, 
lawn-mower, rocket, streetcar, tank, tractor
"""

cifar100_fine_labels=sorted([i.strip() for i in cifar100_fine_labels.split(',')])

cifar100_coarse_labels=['aquatic mammals', 'fish', 'flowers', 'food', 'fruit', 
                       'household electrical devices', 'household furniture',
                       'insects', 'large carnivores', 'large man-made outdoor',
                       'large natural outdoor scenes', 'large omnivores',
                       'medium-sized mammals', 'non-insect invertebrates',
                       'people', 'reptiles', 'small mammals', 'trees', 
                       'vehicles 1', 'vehicles 2']

# setup
analysis_dir = '24-08-2017_07-25-13'

# local directory
#output_dir = path.join('output', analysis_dir)
# for sherlock
output_dir = path.join('/mnt/Sherlock_Scratch/CBMM/output', analysis_dir)

# load data
from keras.datasets import cifar10, cifar100
datasets = {'cifar10': {'labels': cifar10_labels,
                        'data': cifar10.load_data()},
            'cifar100_fine': {'labels': cifar100_fine_labels,
                        'data': cifar100.load_data('fine')},
            'cifar100_coarse': {'labels': cifar100_coarse_labels,
                        'data': cifar100.load_data('coarse')}}
            
        


for dataset in datasets.keys()[0:2]:
    print('*'*80)
    print('Loading %s' % dataset)
    print('*'*80)
    (x_train, y_train), (x_test, y_test) = datasets[dataset]['data']
    (x_train, y_train), (x_val, y_val) = split_val(x_train, y_train)
  
    # comparison images
    compare_labels, compare_data = datasets['cifar100_fine'].values()
    (compare_x_train, compare_y_train), \
        (compare_x_test, compare_y_test) = compare_data
    # reshape data
    num_classes = len(np.unique(y_test))
    y_val = keras.utils.to_categorical(y_val, num_classes)
    
    # load model
    model_file = path.join(output_dir, '%s_model.h5' % dataset)
    with h5py.File(model_file, 'a') as f:
        if 'optimizer_weights' in f.keys():
            del f['optimizer_weights']
    
    # evaluate model
    model = load_model(model_file)
    predictions = model.predict(x_val)
    accuracy = np.mean(np.argmax(predictions,1)==np.argmax(y_val,1))
    
    # extract representations at different levels
    # across stimuli
    layers_of_interest = []
    for layer in model.layers:
        if layer.name.find('conv')!=-1 or layer.name.find('dense')!=-1:
            layers_of_interest.append(layer)
    
    layer_reps = {}
    plt.figure(figsize=(12,8))
    for i, layer in enumerate(layers_of_interest):
        submodel = Model(inputs=model.inputs, outputs=layer.output)
        mean_reps, submodel_reps = get_avg_rep(submodel, 
                                               compare_x_train,
                                               compare_y_train,
                                               10)
        layer_reps[layer.name] = {'mean': mean_reps, 'all': submodel_reps}
        correlation = np.corrcoef(mean_reps)
        plt.subplot(4,2,i+1)
        sns.heatmap(correlation, xticklabels=compare_labels)
    plt.tight_layout()
    
    # extract representations based on weight matrices
    layer_weights = model.layers[1].get_weights()[0]
    plt.imshow(np.sum(layer_weights[:,:,:,0],2))
    
    











