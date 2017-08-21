
from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist

from os import path
import pandas as pd
from sklearn.preprocessing import scale

# load data
data_loc = path.join('Data', 'behavioral_data.csv')

data_df = pd.read_csv(data_loc, index_col=0)
data = data_df.values
n_held_out = data.shape[0]//6
data_train = data[n_held_out:, :]; 
data_held_out = scale(data[:n_held_out,:])

# VAE
batch_size = data_train.shape[0]//5
original_dim = data_train.shape[1]
latent_dim = 2
intermediate_dim = 50
epochs = 50
epsilon_std = 1.0

x = Input(batch_shape=(batch_size, original_dim))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_sigma = Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_sigma) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
# so you could write `Lambda(sampling)([z_mean, z_log_sigma])`
z = Lambda(sampling)([z_mean, z_log_sigma])


decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

# end-to-end autoencoder
vae = Model(x, x_decoded_mean)

# encoder, from inputs to latent space
encoder = Model(x, z_mean)

# generator, from latent space to reconstructed inputs
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

def vae_loss(x, x_decoded_mean):
    xent_loss = metrics.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
    print(xent_loss)
    return xent_loss + kl_loss

vae.compile(optimizer='rmsprop', loss=vae_loss)

vae.fit(data_train, data_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=None)