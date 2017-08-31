# load imports
from glob import glob
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
from utils import convert_to_higher_id, convert_IDs_to_num, load_tiny_imgnet
from utils import get_sample_layer_reps, get_sample_coords, split_val

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import keras
from keras.models import load_model

        
# setup
analysis_dir = 'IMGNET_30-08-2017_15-00-20'
output_dir = path.join('output',analysis_dir)
try:
    makedirs(output_dir)
    makedirs(path.join(output_dir,'Plots'))
except OSError:
    pass
    
# for sherlock
input_dir = path.join('/mnt/Sherlock_Scratch/CBMM/output', analysis_dir)

n_samples = 1000
model_representations = {}
plot=True

# load data
print("Loading Data")
(xtrain, classtrain, bbtrain), \
(xval, classval, bbval), \
(xtest, classtest, bbtest) = load_tiny_imgnet()

# set up sample
sample_index = np.random.choice(range(xtrain.shape[0]), 
                                     n_samples, replace=False)
sample_xtrain = xtrain[sample_index, ::]

# for each model extract representations of compare_dataset for each layer
model_files = glob(path.join(input_dir, '*_model.h5'))
model_distances = {}
model_reps = {}
for i, filey in enumerate(model_files):
    model_name = path.basename(filey).strip('.h5')
    # load model
    with h5py.File(filey, 'a') as f:
        if 'optimizer_weights' in f.keys():
            del f['optimizer_weights']
    model = load_model(filey)
    
    # get representations per layer
    if i == 0:
        sample_coords = get_sample_coords(model, sample_xtrain.shape[0])
    layer_reps, layer_names = get_sample_layer_reps(model, 
                                                   sample_xtrain, 
                                                   sample_coords)
    model_reps[model_name] = layer_reps
    
    # create distance vectors per layer
    layer_distances = []
    for name in layer_names:
        layer_rep = layer_reps[name]
        layer_distances.append(pdist(layer_rep, 'correlation'))
    layer_distances = np.vstack(layer_distances)
    model_distances[model_name] = layer_distances


print('Plotting')

plot_distances = []
for i in range(len(layer_names)):
    distances=pdist(np.vstack(model_distances.values())[i:80:20],'correlation')
    plot_distances.append(squareform(distances))

# plot separate distance graphs for each layer
f = plt.figure(figsize=(30,20))
for i, name in enumerate(layer_names):
    plt.subplot(5,4,i+1)
    if i>15:
        sns.heatmap(plot_distances[i], vmin=0, vmax=1, square=True,
                    cbar=False, xticklabels=model_distances.keys(),
                    yticklabels='')
    else:
        sns.heatmap(plot_distances[i], vmin=0, vmax=1, square=True, 
                    cbar=False, xticklabels='',
                    yticklabels='')
    plt.title(name)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.suptitle('Similarity Across Layers', fontsize=30)

f.savefig(path.join(output_dir, 'Plots', 
                    'sample_layer_model_distances_squareform.png'))

# another visualization of the above
model_combos = list(itertools.combinations(enumerate(model_distances.keys()),2))
f=plt.figure(figsize=(12,24))
sns.heatmap(np.vstack([squareform(i) for i in plot_distances]),
            xticklabels=['%s vs %s' % (i[1], j[1]) for i,j in model_combos],
            yticklabels=layer_names)
plt.xticks(rotation=90, fontsize=20); plt.yticks(rotation=0, fontsize=20)
plt.tight_layout()
f.savefig(path.join(output_dir, 'Plots', 'sample_layer_model_distances.png'))