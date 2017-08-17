from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA, factor_analysis
from sklearn.preprocessing import StandardScaler, scale

data_loc = '/media/Data/Ian/Experiments/expfactory/Self_Regulation_Ontology' \
            + '/Data/Complete_07-08-2017/meaningful_variables_imputed.csv'

data_df = pd.read_csv(data_loc, index_col=0)
data = data_df.values
n_train = int(data.shape[0]*.95)



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
    
def PCA_augmentation(data, reps=5, extrapolate=False, noise_var=.1, lam=.5):
    """
    Use PCA to augment data
    
    Augments data either by replacing final components with noise, or by
    extrapolating between samples in component space
    """
    augmented_data = []
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    pca = PCA()
    rs = pca.fit(scaled_data)
    # augment
    components = rs.components_
    loading = rs.transform(scaled_data)
    # use final dimensions explaining noise_var% of variance
    n_keep = np.sum(np.cumsum(rs.explained_variance_ratio_)<(1-noise_var))
    k = components.shape[0] - n_keep
    if extrapolate==False:
        # replace leftover components with random noise
        for i in range(reps):
            components[n_keep:,:] = np.random.randn(k,components.shape[1])*.05
            augmented_data.append(loading.dot(components))
    else:
        # extrapolate between examples in component space
        reduced_loading = loading[:,:n_keep]
        corr = np.corrcoef(reduced_loading)
        for i, n in enumerate(corr):
            context = reduced_loading[i,:]
            sorted_index = np.argsort(n)[(-1-reps):-1]
            for neighbor in reduced_loading[sorted_index]:
                new = (context-neighbor)*lam + context
                # append noise
                new = np.append(new,np.random.randn(k)*.05)
                # convert back to data space
                new = new.dot(components)
                augmented_data.append(new)
    augmented_data = np.vstack([scaled_data, np.vstack(augmented_data)])
    return augmented_data

def FA_augmentation(data, reps=5, n_components=30, lam=.5):
    """
    Use FA to augment data
    
    Augments data by extrapolating between samples in component space
    """
    augmented_data = []
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    fa = factor_analysis.FactorAnalysis(n_components)
    rs = fa.fit(scaled_data)
    components = rs.components_
    loadings = rs.transform(scaled_data)
    # augment
    new = extrapolate(loadings).dot(components)
    augmented_data = np.vstack([scaled_data, new])
    return augmented_data

def autoencoder_augmentation(encoder, decoder, data, reps=5):
    """
    Use an autodecoder to augment data
    
    Augments data by extrapolating between samples in component space
    """
    encoded_vecs = encoder.predict(data)
    encoded_vecs = extrapolate(encoded_vecs, reps)
    decoded_vecs = decoder.predict(encoded_vecs)
    return decoded_vecs

# ***************************************************************************
# autoencoder
# ***************************************************************************
from keras import regularizers
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam

# this is our input placeholder
input_vec = Input(shape=(data.shape[1],))
def make_autoencoder(input_vec, encoding_dim=100, regularize=False, 
                     dropout=False):
    if regularize:
        encoded = Dense(encoding_dim, activation='relu',
                        W_regularizer=regularizers.l1(.001))(input_vec)
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


autoencoder, encoder, decoder = make_autoencoder(input_vec, 200, regularize=True)
autoencoder.compile(optimizer=Adam(lr=.1, decay=.001), loss='mse')

# select training/test/val
np.random.shuffle(data)
data_train = data[:n_train, :]; np.random.shuffle(data_train)
data_held_out = scale(data[n_train:,:])
n_val = int(data_train.shape[0]*.1)
x_train = scale(data_train[n_val:,:])
x_val = scale(data_train[:n_val,:])
# augment training data
#x_train = PCA_augmentation(x_train, reps=10)
#x_train = FA_augmentation(x_train, reps=10, n_components=50)

# train autoencoder
autoencoder.fit(x_train, x_train,
                epochs=6000,
                batch_size=200,
                shuffle=True,
                validation_data=(x_val, x_val),
                callbacks=[EarlyStopping(min_delta=.001, patience=500),
                           TensorBoard(log_dir='/tmp/autoencoder')])


# visualize decoding
test_data = scale(data_held_out)
n_test = test_data.shape[0]
encoded_imgs = encoder.predict(test_data)
decoded_imgs = decoder.predict(encoded_imgs)
corr = pd.DataFrame(np.vstack([test_data,decoded_imgs])).T.corr().values
np.mean(np.diag(corr[n_test:,:n_test]))
# plot
plt.figure(figsize=(12,8))
sns.heatmap(corr)

# augment data
augmented_data = autoencoder_augmentation(encoder, decoder, scale(data))

# compare augmented data RSA to original data space
plt.figure(figsize=(12,8))
plt.scatter(tril(np.corrcoef(scale(data).T)), 
            tril(np.corrcoef(augmented_data.T)))
plt.xlabel('Original Data Correlations', fontsize=20)
plt.ylabel('Augmented Data Correlations', fontsize=20)