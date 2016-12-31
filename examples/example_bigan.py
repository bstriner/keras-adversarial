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
import os


def model_encoder(latent_dim, input_shape, hidden_dim=1024, reg=lambda: l1(1e-5), batch_norm_mode=0):
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


def model_discriminator(latent_dim, input_shape, output_dim=1, hidden_dim=2048,
                        reg=lambda: l1l2(1e-7, 1e-7), batch_norm_mode=1, dropout=0.5):
    z = Input((latent_dim,))
    x = Input(input_shape, name="x")
    h = merge([z, Flatten()(x)], mode='concat')

    h1 = Dense(hidden_dim, name="discriminator_h1", W_regularizer=reg())
    b1 = BatchNormalization(mode=batch_norm_mode)
    h2 = Dense(hidden_dim, name="discriminator_h2", W_regularizer=reg())
    b2 = BatchNormalization(mode=batch_norm_mode)
    h3 = Dense(hidden_dim, name="discriminator_h3", W_regularizer=reg())
    b3 = BatchNormalization(mode=batch_norm_mode)
    y = Dense(output_dim, name="discriminator_y", activation="sigmoid", W_regularizer=reg())

    # training model uses dropout
    _h = h
    _h = Dropout(dropout)(LeakyReLU(0.2)((b1(h1(_h)))))
    _h = Dropout(dropout)(LeakyReLU(0.2)((b2(h2(_h)))))
    _h = Dropout(dropout)(LeakyReLU(0.2)((b3(h3(_h)))))
    ytrain = y(_h)
    mtrain = Model([z, x], ytrain, name="discriminator_train")

    # testing model does not use dropout
    _h = h
    _h = LeakyReLU(0.2)((b1(h1(_h))))
    _h = LeakyReLU(0.2)((b2(h2(_h))))
    _h = LeakyReLU(0.2)((b3(h3(_h))))
    ytest = y(_h)
    mtest = Model([z, x], ytest, name="discriminator_test")

    return mtrain, mtest


def example_bigan(path, adversarial_optimizer):
    # z \in R^100
    latent_dim = 25
    # x \in R^{28x28}
    input_shape = (28, 28)

    # generator (z -> x)
    generator = model_generator(latent_dim, input_shape)
    # encoder (x ->z)
    encoder = model_encoder(latent_dim, input_shape)
    # autoencoder (x -> x')
    autoencoder = Model(encoder.inputs, generator(encoder(encoder.inputs)))
    # discriminator (x -> y)
    discriminator_train, discriminator_test = model_discriminator(latent_dim, input_shape)
    # bigan (z, x - > yfake, yreal)
    bigan_generator = simple_bigan(generator, encoder, discriminator_test)
    bigan_discriminator = simple_bigan(generator, encoder, discriminator_train)
    # z generated on GPU based on batch dimension of x
    x = bigan_generator.inputs[1]
    z = normal_latent_sampling((latent_dim,))(x)
    # eliminate z from inputs
    bigan_generator = Model([x], fix_names(bigan_generator([z, x]), bigan_generator.output_names))
    bigan_discriminator = Model([x], fix_names(bigan_discriminator([z, x]), bigan_discriminator.output_names))

    generative_params = generator.trainable_weights + encoder.trainable_weights

    # print summary of models
    generator.summary()
    encoder.summary()
    discriminator_train.summary()
    bigan_discriminator.summary()
    autoencoder.summary()

    # build adversarial model
    model = AdversarialModel(player_models=[bigan_generator, bigan_discriminator],
                             player_params=[generative_params, discriminator_train.trainable_weights],
                             player_names=["generator", "discriminator"])
    model.adversarial_compile(adversarial_optimizer=adversarial_optimizer,
                              player_optimizers=[Adam(1e-4, decay=1e-4), Adam(1e-3, decay=1e-4)],
                              loss='binary_crossentropy')

    # load mnist data
    xtrain, xtest = mnist_data()

    # callback for image grid of generated samples
    def generator_sampler():
        zsamples = np.random.normal(size=(10 * 10, latent_dim))
        return generator.predict(zsamples).reshape((10, 10, 28, 28))

    generator_cb = ImageGridCallback(os.path.join(path, "generated-epoch-{:03d}.png"), generator_sampler)

    # callback for image grid of autoencoded samples
    def autoencoder_sampler():
        xsamples = n_choice(xtest, 10)
        xrep = np.repeat(xsamples, 9, axis=0)
        xgen = autoencoder.predict(xrep).reshape((10, 9, 28, 28))
        xsamples = xsamples.reshape((10, 1, 28, 28))
        x = np.concatenate((xsamples, xgen), axis=1)
        return x

    autoencoder_cb = ImageGridCallback(os.path.join(path, "autoencoded-epoch-{:03d}.png"), autoencoder_sampler)

    # train network
    y = gan_targets(xtrain.shape[0])
    ytest = gan_targets(xtest.shape[0])
    history = model.fit(x=xtrain, y=y, validation_data=(xtest, ytest), callbacks=[generator_cb, autoencoder_cb],
                        nb_epoch=100, batch_size=32)

    # save history
    df = pd.DataFrame(history.history)
    df.to_csv(os.path.join(path, "history.csv"))

    # save model
    encoder.save(os.path.join(path, "encoder.h5"))
    generator.save(os.path.join(path, "generator.h5"))
    discriminator_train.save(os.path.join(path, "discriminator.h5"))


def main():
    example_bigan("output/bigan", AdversarialOptimizerSimultaneous())


if __name__ == "__main__":
    main()
