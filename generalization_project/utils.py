from contextlib import closing
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from math import ceil
import numpy as np
from glob import glob
from os import path
from PIL import Image
import tarfile
from xml.etree import ElementTree as etree


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
                            'data': cifar100.load_data('coarse')},
                'cifar100_arbitrary': {'labels': ['group%s' % i for i in range(20)],
                                       'data': arbitrary_load()}}
    return datasets
            
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

   
# imgnet helper functions

def get_imgnet_labels():
    labels=np.genfromtxt('words.txt',dtype='str', delimiter='\t')
    labels = {k:v for k,v in labels}
    return labels

def load_imgnet():
    images = []
    ids = []
    bbs=[]
    bb_labels=[]
    resize_props = []
    for filey in ['n00015388.tar', 'n00007846.tar']:
        tar = tarfile.open("Data/%s" % filey)
        members = tar.getmembers()
        for member in members:
            read_tar=tar.extractfile(member).read()
            from cStringIO import StringIO
            file_jpg = StringIO(read_tar)
            img=Image.open(file_jpg)
            resize_props.append([round(256.0/i,3) for i in img.size])
            img = np.asarray(img.resize((256,256)))
            if img.shape == (256,256,3):
                ids.append(member.name.split('.')[0])
                images.append(img)
        tar.close()
        
        with tarfile.open("Data/%s.gz" % filey) as archive:
            for member in archive:
                if member.isreg() and member.name.endswith('.xml'): # regular xml file
                    with closing(archive.extractfile(member)) as xmlfile:
                        root = etree.parse(xmlfile).getroot()
                        if filey.strip('.tar') in root[1].text:
                            name = root[1].text
                            image_i = ids.index(name)
                            bb = [int(i.text) for i in root[5][4][0:4]]
                            # scale bb based on image size change
                            resize_prop = resize_props[image_i]
                            bb[0]*=resize_prop[0]; bb[1]*=resize_prop[1]
                            bb[2]*=resize_prop[0]; bb[3]*=resize_prop[1]
                            bbs.append(bb)
                            bb_labels.append(name)
    images = np.stack(images,0)
    bbs = np.stack(bbs,0)
    return (images,ids), (bbs, bb_labels)
    
def plt_imgnet(images, ids, bbs, bb_labels):
    semantic_labels = get_imgnet_labels()
    bb_i = np.random.choice(range(len(bb_labels)))
    image_i = ids.index(bb_labels[bb_i])
    image = images[image_i,::]
    bb = bbs[bb_i]
    bb_coords = bb[0:2]
    bb_width = (bb[2]-bb[0])
    bb_height = (bb[3]-bb[1])
    fig,ax = plt.subplots(1)
    ax.imshow(image)
    rect = patches.Rectangle(bb_coords,bb_width,bb_height,
                             linewidth=3,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    plt.title(semantic_labels[ids[image_i]], fontsize=30)

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
                    bbs_file = {a:((b,c,d,e)) for a,b,c,d,e in bbs_file}
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
                