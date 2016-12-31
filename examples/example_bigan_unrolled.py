import matplotlib as mpl

# This line allows mpl to run with no DISPLAY defined
mpl.use('Agg')

from keras.layers import Dense, Reshape, Flatten, Input, merge, Dropout
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.regularizers import l1, l1l2
import keras.backend as K
import pandas as pd
import numpy as np

from keras_adversarial import AdversarialModel, ImageGridCallback, simple_gan, gan_targets, fix_names, n_choice, \
    simple_bigan
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling, AdversarialOptimizerAlternating
from mnist_utils import mnist_data
from example_gan import model_generator
from keras.layers import BatchNormalization, LeakyReLU


def model_encoder(latent_dim, input_shape, hidden_dim=1024, reg=lambda: l1(1e-5), batch_norm_mode=2):
    x = Input(input_shape, name="x")
    h = Flatten()(x)
    h = Dense(hidden_dim, name="encoder_h1", W_regularizer=reg())(h)
    h = BatchNormalization(mode=batch_norm_mode)(h)
    h = LeakyReLU(0.2)(h)
    h = Dense(hidden_dim / 2, name="encoder_h2", W_regularizer=reg())(h)
    h = BatchNormalization(mode=batch_norm_mode)(h)
    h = LeakyReLU(0.2)(h)
    h = Dense(hidden_dim / 4, name="encoder_h3", W_regularizer=reg())(h)
    h = BatchNormalization(mode=batch_norm_mode)(h)
    h = LeakyReLU(0.2)(h)
    mu = Dense(latent_dim, name="encoder_mu", W_regularizer=reg())(h)
    log_sigma_sq = Dense(latent_dim, name="encoder_log_sigma_sq", W_regularizer=reg())(h)
    z = merge([mu, log_sigma_sq], mode=lambda p: p[0] + K.random_normal(p[0].shape) * K.exp(p[1] / 2),
              output_shape=lambda x: x[0])
    return Model(x, z, name="encoder")


def model_discriminator(latent_dim, input_shape, output_dim=1, hidden_dim=1024,
                        reg=lambda: l1l2(1e-4, 1e-4), batch_norm_mode=1):
    z = Input((latent_dim,))
    x = Input(input_shape, name="x")
    h = merge([z, Flatten()(x)], mode='concat')
    h = Dense(hidden_dim, name="discriminator_h1", W_regularizer=reg())(h)
    h = BatchNormalization(mode=batch_norm_mode)(h)
    h = LeakyReLU(0.2)(h)
    h = Dropout(0.5)(h)
    h = Dense(hidden_dim / 2, name="discriminator_h2", W_regularizer=reg())(h)
    h = BatchNormalization(mode=batch_norm_mode)(h)
    h = LeakyReLU(0.2)(h)
    h = Dropout(0.5)(h)
    h = Dense(hidden_dim / 4, name="discriminator_h3", W_regularizer=reg())(h)
    h = BatchNormalization(mode=batch_norm_mode)(h)
    h = LeakyReLU(0.2)(h)
    h = Dropout(0.5)(h)
    y = Dense(output_dim, name="discriminator_y", activation="sigmoid", W_regularizer=reg())(h)
    return Model([z, x], y, name="discriminator")


def main():
    # z \in R^100
    latent_dim = 100
    # x \in R^{28x28}
    input_shape = (28, 28)

    # generator (z -> x)
    generator = model_generator(latent_dim, input_shape)
    # encoder (x ->z)
    encoder = model_encoder(latent_dim, input_shape)
    # autoencoder (x -> x')
    autoencoder = Model(encoder.inputs, generator(encoder(encoder.inputs)))
    # discriminator (x -> y)
    discriminator = model_discriminator(latent_dim, input_shape)
    # bigan (x - > yfake, yreal), z generated on GPU
    bigan = simple_bigan(generator, encoder, discriminator, normal_latent_sampling((latent_dim,)))

    generative_params = generator.trainable_weights + encoder.trainable_weights

    # print summary of models
    generator.summary()
    encoder.summary()
    discriminator.summary()
    bigan.summary()
    autoencoder.summary()

    # build adversarial model
    model = AdversarialModel(base_model=bigan,
                             player_params=[generative_params, discriminator.trainable_weights],
                             player_names=["generator", "discriminator"])
    model.adversarial_compile(adversarial_optimizer=AdversarialOptimizerSimultaneous(),
                              player_optimizers=[Adam(1e-4, decay=1e-4), Adam(1e-3, decay=1e-4)],
                              loss='binary_crossentropy')

    # train model
    xtrain, xtest = mnist_data()

    def generator_sampler():
        zsamples = np.random.normal(size=(10 * 10, latent_dim))
        return generator.predict(zsamples).reshape((10, 10, 28, 28))

    generator_cb = ImageGridCallback("output/bigan/generated-epoch-{:03d}.png", generator_sampler)

    def autoencoder_sampler():
        xsamples = n_choice(xtest, 10)
        xrep = np.repeat(xsamples, 9, axis=0)
        xgen = autoencoder.predict(xrep).reshape((10, 9, 28, 28))
        xsamples = xsamples.reshape((10, 1, 28, 28))
        x = np.concatenate((xsamples, xgen), axis=1)
        return x

    autoencoder_cb = ImageGridCallback("output/bigan/autoencoded-epoch-{:03d}.png", autoencoder_sampler)

    y = gan_targets(xtrain.shape[0])
    ytest = gan_targets(xtest.shape[0])
    history = model.fit(x=xtrain, y=y, validation_data=(xtest, ytest), callbacks=[generator_cb, autoencoder_cb],
                        nb_epoch=100, batch_size=32)
    df = pd.DataFrame(history.history)
    df.to_csv("output/bigan/history.csv")

    encoder.save("output/bigan/encoder.h5")
    generator.save("output/bigan/generator.h5")
    discriminator.save("output/bigan/discriminator.h5")


if __name__ == "__main__":
    main()
