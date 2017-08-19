# see DATASET AUGMENTATION IN FEATURESPACE (2017) for inspiration
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam

from itertools import product
import numpy as np
from os import path
import pandas as pd
import pickle
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale

from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('GTK')
import seaborn as sns

data_loc = path.join('Data', 'meaningful_variables_imputed.csv')

data_df = pd.read_csv(data_loc, index_col=0)
data = data_df.values
n_train = int(data.shape[0]*.95)

np.random.shuffle(data)
data_train = data[:n_train, :]; 
data_held_out = scale(data[n_train:,:])


# ***************************************************************************
# helper functions
# ***************************************************************************

def tril(square_m):
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

def make_autoencoder(input_vec, encoding_dim=100, regularize=None, 
                     dropout=False):
    if regularize:
        encoded = Dense(encoding_dim, activation='sigmoid',
                        kernel_regularizer=regularizers.l1(regularize))(input_vec)
    else:
        encoded = Dense(encoding_dim, activation='relu')(input_vec)
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

def run_autoencoder(train, val, dim, l1, dropout, epochs=100):
    input_vec = Input(shape=(data.shape[1],))
    autoencoder, encoder, decoder = make_autoencoder(input_vec, dim, 
                                                     regularize=l1,
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
                    verbose=0,
                    callbacks=callbacks)
    return out, {'autoencoder': autoencoder, 
                 'encoder': encoder, 
                 'decoder': decoder}


def KF_CV(data, param_space, splits=5):
    KF = KFold(splits)
    folds = list(KF.split(data))
    param_combinations = list(product(param_space['dim'],
                                      param_space['Wl1'],
                                      param_space['dropout']))
    param_scores = {}
    # grid search over params
    for dim,l,dropout in param_combinations:
        print('Testing parameters: %s, %s, %s' % (dim,l,dropout))
        CV_scores = []
        for train_i, val_i in folds:
            x_train = scale(data[train_i,:])
            x_val = scale(data[val_i,:])
            out, models = run_autoencoder(x_train, x_val, dim, l,
                                          dropout, epochs=10000)
            final_score = out.history['val_loss'][-1]
            CV_scores.append(final_score)
            print('CV score: %s' % final_score)
        param_scores[(dim,l,dropout)] = (np.mean(CV_scores))
    best_params = min(param_scores, key=lambda k: param_scores[k])
    return best_params, param_scores
    
# ***************************************************************************
# autoencoder
# ***************************************************************************
param_space = {'dim': [50, 150, 250, 350], 'Wl1': [None, .01, .001, .0001],
               'dropout': [True, False]}
best_params, param_scores = KF_CV(data_train, param_space, splits=4)
pickle.dump(param_scores, open(path.join('output', 
                                         'data_augmentation_CV_results.pkl'), 
                                         'wb'))
out, models = run_autoencoder(scale(data_train), None, best_params[0], 
                              best_params[1], epochs=1000)
for name,m in models.items():
    m.save(path.join('output', 'data_augmentation_%s.h5' % name))
    
# ***************************************************************************
# test
# ***************************************************************************

# visualize decoding
test_data = scale(data_held_out)
n_test = test_data.shape[0]
encoded_imgs = models['encoder'].predict(test_data)
decoded_imgs = models['decoder'].predict(encoded_imgs)
corr = pd.DataFrame(np.vstack([test_data,decoded_imgs])).T.corr().values
np.mean(np.diag(corr[n_test:,:n_test]))


# augment data
augmented_data = autoencoder_augmentation(models['encoder'], 
                                          models['decoder'], 
                                          scale(data))

# save augmented data
augmented_data = pd.DataFrame(augmented_data, columns=data_df.columns)
augmented_data.to_csv(path.join('output','augmented_data.csv'))

# plot
f = plt.figure(figsize=(12,8))
sns.heatmap(corr)
f.savefig(path.join('Plots','test_reconsturction_performance.png'))

# compare augmented data RSA to original data space
f=plt.figure(figsize=(12,8))
plt.scatter(tril(np.corrcoef(scale(data).T)), 
            tril(np.corrcoef(augmented_data.T)))
plt.xlabel('Original Data Correlations', fontsize=20)
plt.ylabel('Augmented Data Correlations', fontsize=20)
f.savefig(path.join('Plots','augmented_data_corr_comparison.png'))

