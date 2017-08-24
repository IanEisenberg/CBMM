from matplotlib import pyplot as plt
import numpy as np

def get_avg_rep(submodel, X, Y, n_exemplars):
        reps = []
        n_classes = len(Y)
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
    
def plot_cifar10(X,Y):
    num_classes = len(Y[0,:])
    # plot random images from each class
    class_names = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']
    fig = plt.figure(figsize=(8,3))
    for i in range(num_classes):
        ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
        idx = np.where(X[:]==i)[0]
        features_idx = Y[idx,::]
        img_num = np.random.randint(features_idx.shape[0])
        im = features_idx[img_num,::]
        ax.set_title(class_names[i])
        plt.imshow(im)
    plt.show()

def split_val(x_train, y_train):
    split = len(x_train)*4//5
    (x_train, y_train), (x_val, y_val) = (x_train[:split], y_train[:split]), \
                                         (x_train[split:], y_train[split:])
    return (x_train, y_train), (x_val, y_val)