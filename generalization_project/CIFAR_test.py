# load imports
import h5py
from math import ceil
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from os import path, makedirs
import pickle
import seaborn as sns
from utils import get_datasets, get_layer_reps, split_val

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

compare_dataset = 'cifar100_fine'
n_exemplars = 100
model_representations = {}
plot=True

# comparison images
compare_labels, compare_data = datasets[compare_dataset].values()
(compare_x_train, compare_y_train), \
(compare_x_test, compare_y_test) = compare_data
num_compare_classes = len(np.unique(compare_y_test))

# for each dataset extract representations of compare_dataset for each layer
for i, dataset in enumerate(datasets.keys()):
    print('*'*80)
    print('Loading %s' % dataset)
    print('*'*80)
    (x_train, y_train), (x_test, y_test) = datasets[dataset]['data']
    (x_train, y_train), (x_val, y_val) = split_val(x_train, y_train)

    # reshape data
    num_classes = len(np.unique(y_test))
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
    
    # evaluate model
    print('Calculating final x_val on validation')
    predictions = model.predict(x_val)
    accuracy = np.mean(np.argmax(predictions,1)==np.argmax(y_val,1))
    
    # extract representations at different levels
    # across stimuli
    print("Getting Representations")
    rep_loc = path.join(output_dir,'%s_compare_%s_%sreps.pkl'% 
                                    (dataset,compare_dataset,n_exemplars))
    if path.exists(rep_loc):
        layer_reps, layer_names = pickle.load(open(rep_loc,'rb'))
    else:
        layer_reps, layer_names = get_layer_reps(model, compare_x_train, 
                                                 compare_y_train, n_exemplars)
        pickle.dump((layer_reps, layer_names), open(rep_loc,'wb'))
    model_representations[dataset] = layer_reps
    
    print("Plotting")
    if i==0:
        # create ordering based on last layer before readout:
        last_reps = layer_reps[layer_names[-1]]['flat_mean']
        linkage_mat = linkage(pdist(last_reps, metric='cosine'))
        leaves = dendrogram(linkage_mat, no_plot=True)['leaves']
        reordered_labels = [compare_labels[i] for i in leaves]  
    if plot==True:
        # plot represention heatmpa
        f = plt.figure(figsize=(60,50))
        for i, name in enumerate(layer_names):
            # get mean_reps for layer and reorder
            mean_reps = layer_reps[name]['flat_mean'][leaves]
            plt.subplot(ceil(len(layer_names)/2.0),2,i+1)
            correlation = np.corrcoef(mean_reps)
            sns.heatmap(correlation, xticklabels='', square=True)
            plt.yticks(range(len(compare_labels)), reordered_labels, 
                       fontsize=6+(100/num_compare_classes), rotation=0)
            plt.title(name, fontsize=20)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.suptitle('%s on %s - accuracy:%s%%' % (dataset, compare_dataset, 
                                                   accuracy),
                     fontsize = 30, y=.95)
        f.savefig(path.join(output_dir,
                            'Plots',
                            '%s_compare_%s_layer_representations.png' \
                            % (dataset, compare_dataset)))
    
    

overall_reps = []
labels = []
rep_layers = layer_names[1:]
nl = len(rep_layers) # number of layers
colors = sns.color_palette('Reds',nl) \
         + sns.color_palette('Blues',nl) \
         + sns.color_palette('Greens',nl) \
         + sns.color_palette('Oranges',nl)
         
for model, val in model_representations.items()[0:3]:
    for layer in rep_layers:
        labels.append(model+'_'+layer)
        overall_reps.append(squareform(val[layer]['flat_mean']))
overall_reps = np.vstack(overall_reps)

pca = PCA(4)
reduced=pca.fit_transform(overall_reps)

# clustermap of correlation distances
f=sns.clustermap(squareform(pdist(overall_reps,'correlation')), 
               xticklabels=labels, yticklabels=labels)
f.savefig(path.join(output_dir,'Plots','compare_%s_clustermap.png' \
                    % (compare_dataset)))

# distance plots, not saved
f = plt.figure(figsize=(12,8))
for i in range(nl):
    plt.plot(squareform(pdist(overall_reps,'correlation'))[:nl,i],
             label=labels[i], linewidth=3)
plt.legend()

f = plt.figure(figsize=(12,8))
for i in range(nl):
    plt.plot(squareform(pdist(overall_reps,'correlation'))[nl:,i],
            label=labels[i], linewidth=3)
plt.legend()

# 3d scatter
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection='3d')
Axes3D.scatter(ax,reduced[:nl,0], reduced[:nl,1], 
                          reduced[:nl,2], c='red',
               s=[i**2.5*6 for i in range(nl)], 
               label=model_representations.keys()[0])
Axes3D.scatter(ax,reduced[nl:nl*2,0], reduced[nl:nl*2,1], 
                          reduced[nl:nl*2,2], c='blue',
               s=[i**2.5*6 for i in range(nl)],
               label=model_representations.keys()[1])
Axes3D.scatter(ax,reduced[nl*2:nl*3,0], reduced[nl*2:nl*3,1], 
                          reduced[nl*2:nl*3,2], c='green',
               s=[i**2.5*6 for i in range(nl)],
               label=model_representations.keys()[2])
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')
plt.legend()
fig.savefig(path.join(output_dir,'Plots','compare_%s_3Dplot.png' \
                    % (compare_dataset)))

# save RSA in pixel space
avg_imgs = []
for i  in range(num_compare_classes):
    img_indices = np.where([j[0]==i for j in compare_y_train])[0]
    avg_img = np.mean(compare_x_train[img_indices], 0)
    avg_imgs.append(avg_img)
avg_imgs = np.stack(avg_imgs,0).reshape((num_compare_classes, 32*32*3))['leaves']
f = plt.figure(figsize=(12,8))
sns.heatmap(np.corrcoef(avg_imgs))
f.savefig(path.join(output_dir,'Plots','similarity_raw_img_space.png'))