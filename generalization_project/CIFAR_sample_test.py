# load imports
import h5py
import itertools
from math import ceil
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import numpy as np
from os import path, makedirs
import pickle
import seaborn as sns
from utils import get_datasets, get_sample_layer_reps, get_sample_coords, split_val

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

import keras
from keras.models import load_model

datasets = get_datasets()
        
# setup
analysis_dir = '26-08-2017_14-07-46'
output_dir = path.join('output',analysis_dir)
try:
    makedirs(output_dir)
    makedirs(path.join(output_dir,'Plots'))
except OSError:
    pass
    
# local directory
#input_dir = path.join('output', analysis_dir)
# for sherlock
input_dir = path.join('/mnt/Sherlock_Scratch/CBMM/output', analysis_dir)

compare_dataset = 'cifar100_coarse'
compare_samples = 1000
model_representations = {}
plot=True

# comparison images
compare_labels, compare_data = datasets[compare_dataset].values()
(compare_x_train, compare_y_train), \
(compare_x_test, compare_y_test) = compare_data
compare_index = np.random.choice(range(compare_x_train.shape[0]), 
                                 compare_samples, replace=False)
compare_x_train = compare_x_train[compare_index,::]
compare_y_train = compare_y_train[compare_index,::]

# get labels for comparison images in coarse space

# for each dataset extract representations of compare_dataset for each layer
model_distances = {}
model_reps = {}
for i, dataset in enumerate(datasets.keys()):
    print('*'*80)
    print('Loading %s' % dataset)
    print('*'*80)
    (x_train, y_train), (x_test, y_test) = datasets[dataset]['data']
    (x_train, y_train), (x_val, y_val) = split_val(x_train, y_train)
  
    # reshape data
    num_classes = len(np.unique(y_test))
    num_compare_classes = len(np.unique(compare_y_test))
    y_val = keras.utils.to_categorical(y_val, num_classes)
    
    # load model
    model_file = path.join(input_dir, '%s_model.h5' % dataset)
    if not path.exists(model_file):
        print("%s wasn't found" % model_file)
        continue
    with h5py.File(model_file, 'a') as f:
        if 'optimizer_weights' in f.keys():
            del f['optimizer_weights']
    model = load_model(model_file)
    
    # get representations per layer
    if i == 0:
        sample_coords = get_sample_coords(model, compare_x_train.shape[0])
    layer_reps, layer_names = get_sample_layer_reps(model, 
                                                   compare_x_train, 
                                                   sample_coords)
    model_reps[dataset] = layer_reps
    
    # create distance vectors per layer
    layer_distances = []
    for name in layer_names:
        layer_rep = layer_reps[name]
        layer_distances.append(pdist(layer_rep, 'correlation'))
    layer_distances = np.vstack(layer_distances)
    model_distances[dataset] = layer_distances


print('Plotting')
pca = PCA(2)
color_palette = sns.color_palette('hls',20)
colors = [color_palette[i[0]] for i in compare_y_train]
f = plt.figure(figsize=(30,20))
for i, name in enumerate(layer_names):
    rep = layer_reps[name]
    pca.fit(rep)
    PCA_rep = pca.transform(rep)
    plt.subplot(5,4,i+1)
    plt.scatter(PCA_rep[:,0], PCA_rep[:,1], c=colors)
    plt.title(name,fontsize=20)


plot_distances = []
for i in range(len(layer_names)):
    plot_distances.append(squareform(pdist(np.vstack(model_distances.values())[i:80:20],'correlation')))

# plot separate distance graphs for each layer
f = plt.figure(figsize=(30,20))
for i, name in enumerate(layer_names):
    plt.subplot(5,4,i+1)
    if i>15:
        sns.heatmap(plot_distances[i], vmin=0, vmax=1, square=True,
                    cbar=False, xticklabels=datasets.keys(),
                    yticklabels='')
    else:
        sns.heatmap(plot_distances[i], vmin=0, vmax=1, square=True, 
                    cbar=False, xticklabels='',
                    yticklabels='')
    plt.title(name)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.suptitle('Similarity Across Layers', fontsize=30)

f.savefig(path.join(output_dir, 'Plots', 'sample_layer_model_distances_squareform.png'))

# another visualization of the above
model_combos = list(itertools.combinations(enumerate(model_distances.keys()),2))
f=plt.figure(figsize=(12,24))
sns.heatmap(np.vstack([squareform(i) for i in plot_distances]),
            xticklabels=['%s vs %s' % (i[1], j[1]) for i,j in model_combos],
            yticklabels=layer_names)
plt.xticks(rotation=90, fontsize=20); plt.yticks(rotation=0, fontsize=20)
plt.tight_layout()
f.savefig(path.join(output_dir, 'Plots', 'sample_layer_model_distances.png'))