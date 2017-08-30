from matplotlib import pyplot as plt
import matplotlib.patches as patches
from math import ceil
import numpy as np
from glob import glob
from os import path
from PIL import Image


from keras.models import Model
from keras.datasets import cifar10, cifar100

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
        reps.append(rep)
    # avg vector for each exemplar
    mean_reps = np.array([np.mean(r,0) for r in reps])
    return mean_reps, reps

def get_layer_reps(model, X, Y, n_exemplars):
    layers_of_interest = []
    for layer in model.layers:
        if (layer.name.find('conv')!=-1 or layer.name.find('dense')!=-1
            or layer.name.find('readout')!=-1):
            layers_of_interest.append(layer)
    layer_names = [i.name for i in layers_of_interest]
    layer_reps = {}
    for i, layer in enumerate(layers_of_interest):
        submodel = Model(inputs=model.inputs, outputs=layer.output)
        mean_reps, submodel_reps = get_avg_rep(submodel, 
                                               X,
                                               Y,
                                               n_exemplars)
        flattened_shape = (mean_reps.shape[0], np.product(mean_reps.shape[1:]))
        layer_reps[layer.name] = {'flat_mean': np.reshape(mean_reps, 
                                          flattened_shape),
                                    'mean': mean_reps}
    return layer_reps, layer_names

def get_sample_coords(model, samples):
    sample_coords = []
    for s in range(samples):
        coords = {}
        shapes = [(l.name,l.output_shape[1:]) for l in model.layers]
        for name, shape in shapes:
            if len(shape)==3:
                index = [np.random.randint(0,i) for i in shape[:2]]
                coords[name] = index
        sample_coords.append(coords)
    return sample_coords
    
    
def get_sample_layer_reps(model, sample_images, sample_coords):
    # get coordinates for each sample to extract representation
    # extract layers of interest
    layers_of_interest = []
    for layer in model.layers:
        if (layer.name.find('dropout')==-1 and layer.name.find('input')==-1
            and layer.name.find('flatten')==-1):
            layers_of_interest.append(layer)
    # get representations
    layer_names = [i.name for i in layers_of_interest]
    layer_reps = {}
    for i, layer in enumerate(layers_of_interest):
        # get representations for this layer
        submodel = Model(inputs=model.inputs, outputs=layer.output)
        sample_reps = submodel.predict(sample_images)
        # sample one receptive field per sample
        if len(sample_reps.shape) == 4:
            # get coords for layer
            coords = [i[layer.name] for i in sample_coords]
            RF_reps = []
            for i, rep in enumerate(sample_reps):
                RF_rep = rep[coords[i][0],coords[i][1],:]
                RF_reps.append(RF_rep)
        else:
            RF_reps = sample_reps
        layer_reps[layer.name] = np.vstack(RF_reps)
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

#****************************************************************************
    # Dataset helper functions
##****************************************************************************

def arbitrary_load():
    (x_train, y_train), (x_test, y_test) = cifar100.load_data('fine')
    arbitrary_classes = {i:i//5 for i in range(100)}
    y_train = np.array([[arbitrary_classes[i[0]]] for i in y_train])
    y_test = np.array([[arbitrary_classes[i[0]]] for i in y_test])
    return (x_train, y_train), (x_test, y_test)

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
                            'data': cifar100.load_data('coarse')},
                'cifar100_arbitrary': {'labels': ['group%s' % i for i in range(20)],
                                       'data': arbitrary_load()}}
    return datasets


# tiny imgnet helper functions

def convert_IDs_to_num(Y):
    lookup = {val: i for i,val in enumerate(sorted(np.unique(Y)))}
    return [lookup[y] for y in Y], lookup
    
def load_tiny_imgnet_labels():
    labels=np.genfromtxt('words.txt',dtype='str', delimiter='\t')
    labels = {k:v for k,v in labels}
    return labels

def get_tiny_imgnet_label(labels,ID):
    return labels[ID.split('_')[0]]

def load_tiny_imgnet():
    data={}
    for imgset in ['train','val','test']:
        images = []
        ids = []
        bbs=[]
        bbs_file=None
        if imgset == 'train':
            folders = glob('Data/tiny-imagenet-200/%s/*' % imgset)
        else:
            folders = glob('Data/tiny-imagenet-200/%s' % imgset)
        for folder in folders:
            name = path.basename(folder)
            if imgset != 'test':
                bbs_file = np.genfromtxt(path.join(folder,"%s_boxes.txt" % name), 
                                    dtype='str')
                if imgset == 'val':
                    bbs_file = {a:((b,c,d,e),ID) for a,ID,b,c,d,e in bbs_file}
                else:
                    bbs_file = {a:((b,c,d,e),) for a,b,c,d,e in bbs_file}
                    ID = path.basename(folder)
            for image_f in glob(path.join(folder, 'images/*')):
                if bbs_file is not None:
                    bbs.append(bbs_file[path.basename(image_f)][0])
                if imgset == 'val':
                    ID = bbs_file[path.basename(image_f)][1]
                ids.append(ID)
                img = Image.open(image_f)
                img = np.array(img)
                # convert grayscale into rgb
                if len(img.shape)==2:
                    img = np.stack((img,)*3,2)
                images.append(img)
        images = np.stack(images,0)
        if bbs_file is not None:
            bbs = np.stack(bbs,0).astype(int)
        data[imgset] = {'images': images,
                        'ids': ids,
                        'bbs': bbs}
    xtrain = data['train']['images']
    xval = data['val']['images']
    xtest = data['test']['images']
    classtrain = data['train']['ids']
    classval = data['val']['ids']
    classtest = data['test']['ids']
    bbtrain = data['train']['bbs']
    bbval = data['val']['bbs']
    bbtest = data['test']['bbs']
    return (xtrain, classtrain, bbtrain), \
            (xval, classval, bbval), \
            (xtest, classtest, bbtest)
    
def plt_tiny_imgnet(images, ids, bbs):
    labels = load_tiny_imgnet_labels()
    i = np.random.choice(range(len(images)))
    image = images[i,::]
    bb = bbs[i]
    bb_coords = bb[0:2]
    bb_width = (bb[2]-bb[0])
    bb_height = (bb[3]-bb[1])
    fig,ax = plt.subplots(1)
    ax.imshow(image)
    rect = patches.Rectangle(bb_coords,bb_width,bb_height,
                             linewidth=3,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    plt.title(get_tiny_imgnet_label(labels, ids[i]), fontsize=30)              
                