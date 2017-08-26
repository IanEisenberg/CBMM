from matplotlib import pyplot as plt
from math import ceil
import numpy as np

from keras.models import Model
from keras.datasets import cifar10, cifar100

def arbitrary_load():
    (x_train, y_train), (x_test, y_test) = cifar100.load_data('fine')
    arbitrary_classes = {i:i//5 for i in range(100)}
    y_train = np.array([[arbitrary_classes[i[0]]] for i in y_train])
    y_test = np.array([[arbitrary_classes[i[0]]] for i in y_test])
    return (x_train, y_train), (x_test, y_test)
    
def get_avg_rep(submodel, X, Y, n_exemplars):
    """
    Return the average output of a submodel for each unique class in Y. 
    n_exemplars determines the number of images to average over
    """
    reps = []
    n_classes = len(np.unique(Y))
    for class_i in range(n_classes):
        class_index = np.where(Y.flatten()==class_i)[0][:n_exemplars]
        class_images = X[class_index,:]
        rep = submodel.predict(class_images)
        # reshape
        rep = np.reshape(rep, (n_exemplars,np.product(rep.shape[1:])))
        reps.append(rep)
    # avg vector for each exemplar
    mean_reps = np.array([np.mean(r,0) for r in reps])
    return mean_reps, reps

def get_datasets():
    
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
    # load data
    datasets = {'cifar10': {'labels': cifar10_labels,
                            'data': cifar10.load_data()},
                'cifar100_fine': {'labels': cifar100_fine_labels,
                            'data': cifar100.load_data('fine')},
                'cifar100_coarse': {'labels': cifar100_coarse_labels,
                            'data': cifar100.load_data('coarse')}}
    return datasets
            
def get_layer_reps(model, X, Y, n_exemplars):
    layers_of_interest = []
    for layer in model.layers:
        if layer.name.find('conv')!=-1 or layer.name.find('dense')!=-1:
            layers_of_interest.append(layer)
    layer_names = [i.name for i in layers_of_interest]
    layer_reps = {}
    for i, layer in enumerate(layers_of_interest):
        submodel = Model(inputs=model.inputs, outputs=layer.output)
        mean_reps, submodel_reps = get_avg_rep(submodel, 
                                               X,
                                               Y,
                                               n_exemplars)
        layer_reps[layer.name] = {'mean': mean_reps}
    return layer_reps, layer_names
        
def plot_cifar10(X,Y,labels):
    """
    Plots images from dataset
    """
    num_classes = len(labels)
    # plot random images from each class
    fig = plt.figure(figsize=(8,3))
    for i in range(num_classes):
        ax = fig.add_subplot(ceil(num_classes/5.0), 5, 1 + i, xticks=[], yticks=[])
        idx = np.where(X[:]==i)[0]
        features_idx = Y[idx,::]
        img_num = np.random.randint(features_idx.shape[0])
        im = features_idx[img_num,::]
        ax.set_title(labels[i])
        plt.imshow(im)
    plt.show()

def split_val(x_train, y_train):
    "Split training set such that 1/5 is a validation set"
    split = len(x_train)*4//5
    (x_train, y_train), (x_val, y_val) = (x_train[:split], y_train[:split]), \
                                         (x_train[split:], y_train[split:])
    return (x_train, y_train), (x_val, y_val)