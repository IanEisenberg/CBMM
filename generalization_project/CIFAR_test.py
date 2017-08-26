# load imports
import h5py
from math import ceil
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import numpy as np
from os import path, makedirs
import pickle
import seaborn as sns
from utils import get_datasets, get_layer_reps, split_val

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform

import keras
from keras.models import load_model

datasets = get_datasets()
        
# setup
analysis_dir = '24-08-2017_17-26-17'
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

# for each dataset extract representations of compare_dataset for each layer
for i, dataset in enumerate(datasets.keys()):
    print('*'*80)
    print('Loading %s' % dataset)
    print('*'*80)
    (x_train, y_train), (x_test, y_test) = datasets[dataset]['data']
    (x_train, y_train), (x_val, y_val) = split_val(x_train, y_train)
  
    # comparison images
    compare_labels, compare_data = datasets[compare_dataset].values()
    (compare_x_train, compare_y_train), \
        (compare_x_test, compare_y_test) = compare_data
    # reshape data
    num_classes = len(np.unique(y_test))
    y_val = keras.utils.to_categorical(y_val, num_classes)
    
    # load model
    model_file = path.join(input_dir, '%s_model.h5' % dataset)
    with h5py.File(model_file, 'a') as f:
        if 'optimizer_weights' in f.keys():
            del f['optimizer_weights']
    model = load_model(model_file)
    
    # evaluate model
    print('Calculating final accuracy on validation')
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
        last_reps = layer_reps[layer_names[-1]]['mean']
        linkage_mat = linkage(pdist(last_reps, metric='cosine'))
        leaves = dendrogram(linkage_mat, no_plot=True)['leaves']
        reordered_labels = [compare_labels[i] for i in leaves]  
    
    # plot represention heatmpa
    f = plt.figure(figsize=(60,50))
    for i, name in enumerate(layer_names):
        # get mean_reps for layer and reorder
        mean_reps = layer_reps[name]['mean'][leaves]
        plt.subplot(ceil(len(layer_names)/2.0),2,i+1)
        correlation = np.corrcoef(mean_reps)
        sns.heatmap(correlation, xticklabels='', square=True)
        plt.yticks(range(len(compare_labels)), reordered_labels, 
                   fontsize=8+(100/num_classes), rotation=0)
        plt.title(name, fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle('%s on %s - accuracy:%s%%' % (dataset, compare_dataset, accuracy),
                 fontsize = 30, y=.95)
    f.savefig(path.join(output_dir,
                        'Plots',
                        '%s_compare_%s_layer_representations.png' \
                        % (dataset, compare_dataset)))
    
    
    # extract representations based on weight matrices
    layer_weights = model.layers[1].get_weights()[0]
    plt.imshow(np.sum(layer_weights[:,:,:,0],2))
    
    
def tril(mat):
    return mat[np.tril_indices_from(mat, -1)]

overall_reps = []
labels = []
rep_layers = layer_names[1:]
colors = sns.color_palette('Reds',len(rep_layers)) \
         + sns.color_palette('Greens',len(rep_layers)) \
         + sns.color_palette('Blues',len(rep_layers))
         
for model, val in model_representations.items():
    for layer in rep_layers:
        labels.append(model+'_'+layer)
        overall_reps.append(tril(val[layer]['mean']))
overall_reps = np.vstack(overall_reps)

from sklearn.cluster import k_means
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
pca = PCA(2)
tsne = TSNE(2)
reduced=tsne.fit_transform(overall_reps)
reduced=pca.fit_transform(overall_reps)

plt.figure(figsize=(12,8))
plt.scatter(reduced[:,0], reduced[:,1], c=colors, s=150)

sns.clustermap(squareform(pdist(overall_reps,'cosine')), 
               xticklabels=labels, yticklabels=labels)






