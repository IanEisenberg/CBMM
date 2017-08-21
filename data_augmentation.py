# see DATASET AUGMENTATION IN FEATURESPACE (2017) for inspiration
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Dropout, Lambda
from keras.models import Model
from keras.optimizers import Adam

from itertools import product
import numpy as np
from os import path
import pandas as pd
import pickle
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import seaborn as sns

# load data
data_loc = path.join('Data', 'behavioral_data.csv')

data_df = pd.read_csv(data_loc, index_col=0)
data = data_df.values
n_held_out = 50
data_train = data[n_held_out:, :]; 
data_held_out = scale(data[:n_held_out,:])
        
# ***************************************************************************
# helper functions
# ***************************************************************************
def bootstrap_corr(data, test_size, reps=100):
    """
    Bootstraps comparison of 2nd order correlation between large dataset and
    small sample from it
    """
    boot_corrs = []
    orig_corr = np.corrcoef(data.T)
    for rep in range(reps):
        boot_rows = np.random.choice(range(data.shape[0]),test_size)
        boot_data = data[boot_rows,:]
        corr = np.corrcoef(boot_data.T)
        boot_corrs.append(np.corrcoef(tril(corr), tril(orig_corr))[1,0])
    return boot_corrs
    
def tril(square_m):
    """ Returns the lower triangle of a matrix as a vector """
    return square_m[np.tril_indices_from(square_m,-1)]

def extrapolate(feature_data, n_neighbors=5, lam=.5):
    extrapolations = feature_data.tolist()
    distances = np.zeros([feature_data.shape[0]]*2)
    for i, context in enumerate(feature_data):
        temp = feature_data-context
        distance = np.linalg.norm(temp, axis=1)
        distances[i,:] = distance
        sorted_index = np.argsort(distance)[1:(n_neighbors+1)]
        for neighbor in feature_data[sorted_index]:
            new = (context-neighbor)*lam + context
            extrapolations.append(new)
    return np.vstack(extrapolations)
    
def autoencoder_augmentation(encoder, decoder, data, reps=5):
    """
    Use an autodecoder to augment data
    
    Augments data by extrapolating between samples in component space
    """
    encoded_vecs = encoder.predict(data)
    encoded_vecs = extrapolate(encoded_vecs, reps)
    decoded_vecs = decoder.predict(encoded_vecs)
    return decoded_vecs

def make_autoencoder(input_vec, encoding_dim=100, wregularize=0, 
                     aregularize=0, input_noise=0, dropout=False):
    """
    Creates an autoencoder and its corresponding encoder and decoder
    """
    corrupted  = Dropout(input_noise)(input_vec)
    encoded = Dense(encoding_dim, activation='sigmoid',
                    kernel_regularizer=regularizers.l1(wregularize),
                    activity_regularizer=regularizers.l1(aregularize)
                    )(corrupted)
    if dropout:
        encoded = Dropout(.2)(encoded)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(int(input_vec.shape[1]))(encoded)
    autoencoder = Model(input_vec, decoded)
    # create encoder
    encoder = Model(input_vec, encoded)
    # and decoder
    encoded_input = Input(shape=(encoding_dim,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(encoded_input, decoder_layer(encoded_input))
    return autoencoder, encoder, decoder

def run_autoencoder(train, val, params, epochs=10000, verbose=0):
    """
    Trains above autoencoder
    """
    # unpack params
    dim = params['dim']
    al1 = params.get('al1', 0)
    wl1 = params.get('wl1', 0)
    dropout = params.get('dropout', False)
    input_noise = params.get('input_noise', 0)
    
    input_vec = Input(shape=(data.shape[1],))
    autoencoder, encoder, decoder = make_autoencoder(input_vec, dim, 
                                                     wregularize=wl1,
                                                     aregularize=al1,
                                                     input_noise=input_noise,
                                                     dropout=dropout)
    autoencoder.compile(optimizer=Adam(lr=.1, decay=.001), loss='mse')
    if val is not None:
        val=(val,val)
        callbacks = [EarlyStopping(min_delta=.001, patience=500)]
    else:
        callbacks = []
    # train autoencoder
    out = autoencoder.fit(train, train,
                    epochs=epochs,
                    batch_size=epochs,
                    shuffle=True,
                    validation_data=val,
                    verbose=verbose,
                    callbacks=callbacks)
    return out, {'autoencoder': autoencoder, 
                 'encoder': encoder, 
                 'decoder': decoder}


def KF_CV(data, param_space, splits=5, epochs=10000):
    """
    Explores a parameter space using gridsearch using CV
    """
    KF = KFold(splits)
    folds = list(KF.split(data))[:2]
    param_combinations = [dict(zip(param_space, v)) 
                            for v in product(*param_space.values())]
    # grid search over params
    for params in param_combinations:
        print(params)
        CV_scores = []
        for train_i, val_i in folds:
            x_train = scale(data[train_i,:])
            x_val = scale(data[val_i,:])
            out, models = run_autoencoder(x_train, x_val, params, epochs=epochs)
            final_score = out.history['val_loss'][-1]
            CV_scores.append(final_score)
            print('CV score: %s' % final_score)
        params['score'] = np.mean(CV_scores)
    best_params = min(param_combinations, key=lambda k: k['score'])
    return best_params, param_combinations
    
# ***************************************************************************
# autoencoder
# ***************************************************************************
epochs = 2000
param_space = {'dim': [50, 150, 250, 350], 'wl1': [0],
               'al1': [0, .01, .001, .0001], 'input_noise': [0,.2],
               'dropout': [False]}

best_params, param_scores = KF_CV(data_train, param_space, 
                                  splits=5, epochs=epochs)
pickle.dump(param_scores, 
            open(path.join('output', 
                           'data_augmentation_CV_results_%se.pkl' % epochs), 
                            'wb'))
out, models = run_autoencoder(scale(data_train), None, best_params, 
                              epochs=epochs, verbose=1)
    
# ***************************************************************************
# test and validate
# ***************************************************************************
# visualize decoding
test_data = scale(data_held_out)
n_test = test_data.shape[0]
encoded_imgs = models['encoder'].predict(test_data)
decoded_imgs = models['decoder'].predict(encoded_imgs)
corr = pd.DataFrame(np.vstack([test_data,decoded_imgs])).T.corr().values
np.mean(np.diag(corr[n_test:,:n_test]))

# test data augmentation
augmented_data = autoencoder_augmentation(models['encoder'], 
                                          models['decoder'], 
                                          scale(data))
    
# plot
f = plt.figure(figsize=(12,8))
sns.heatmap(corr)
f.savefig(path.join('Plots','test_reconsturction_performance.png'))

# compare augmented data RSA to original data space
boot_corrs = bootstrap_corr(data_train, test_data.shape[0], reps=1000)
augmented_corr = np.corrcoef(tril(np.corrcoef(scale(data_held_out).T)), 
                             tril(np.corrcoef(augmented_data.T)))[0,1]
f = plt.figure(figsize=(12,8))
plt.hist(boot_corrs, bins=50)
ax = plt.gca(); ylim = ax.get_ylim()
plt.vlines(augmented_corr, ylim[0], ylim[1])
f.savefig(path.join('Plots','bootstrap_corr_hist.png'))

f=plt.figure(figsize=(12,8))
plt.scatter(tril(np.corrcoef(scale(data_held_out).T)), 
            tril(np.corrcoef(augmented_data.T)))
plt.xlabel('Original Data Correlations', fontsize=20)
plt.ylabel('Augmented Data Correlations', fontsize=20)
f.savefig(path.join('Plots','augmented_data_corr_comparison.png'))

# ***************************************************************************
# augmented_data
# ***************************************************************************

out, models = run_autoencoder(scale(data), None, best_params,
                              epochs=epochs, verbose=1)
for name,m in models.items():
    m.save(path.join('output', 'data_augmentation_%s.h5' % name))
    
# augment data
augmented_data = autoencoder_augmentation(models['encoder'], 
                                          models['decoder'], 
                                          scale(data))

# save augmented data
augmented_data = pd.DataFrame(augmented_data, columns=data_df.columns)
augmented_data.to_csv(path.join('output','augmented_data.csv'))
