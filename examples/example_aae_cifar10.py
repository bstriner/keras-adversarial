# import os
# os.environ["THEANO_FLAGS"] = "mode=FAST_COMPILE,device=cpu,floatX=float32"

import matplotlib as mpl

# This line allows mpl to run with no DISPLAY defined
mpl.use('Agg')

from keras.layers import Dense, Reshape, Flatten, Dropout, LeakyReLU, Activation, BatchNormalization, SpatialDropout2D
from keras.layers import Input, merge
from keras.layers.convolutional import Convolution2D, UpSampling2D, MaxPooling2D, AveragePooling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.regularizers import l1, l1l2
import keras.backend as K
import pandas as pd
import numpy as np

from keras_adversarial import AdversarialModel, ImageGridCallback, simple_gan, gan_targets, fix_names, n_choice, \
    simple_bigan
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling, AdversarialOptimizerAlternating
from cifar10_utils import cifar10_data
from keras.layers import BatchNormalization, LeakyReLU, Activation
from image_utils import dim_ordering_fix, dim_ordering_unfix, dim_ordering_shape, channel_axis
import os


def model_generator(latent_dim, nch = 512, dropout=0.5, reg = lambda: l1l2(l1=1e-7, l2=1e-7)):
    model = Sequential(name="decoder")
    h = 5
    model.add(Dense(input_dim=latent_dim, output_dim=nch * 4 * 4, W_regularizer=reg()))
    model.add(Reshape(dim_ordering_shape((nch, 4, 4))))
    model.add(SpatialDropout2D(dropout))
    model.add(LeakyReLU(0.2))
    model.add(Convolution2D(nch / 2, h, h, border_mode='same', W_regularizer=reg()))
    model.add(SpatialDropout2D(dropout))
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(nch / 2, h, h, border_mode='same', W_regularizer=reg()))
    model.add(SpatialDropout2D(dropout))
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(nch / 4, h, h, border_mode='same', W_regularizer=reg()))
    model.add(SpatialDropout2D(dropout))
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(3, h, h, border_mode='same', W_regularizer=reg()))
    model.add(Activation('sigmoid'))
    return model


def model_encoder(latent_dim, input_shape, nch=512, reg=lambda: l1l2(l1=1e-7, l2=1e-7), dropout=0.5):
    k = 5
    x = Input(input_shape)
    h = Convolution2D(nch / 4, k, k, border_mode='same', W_regularizer=reg())(x)
    h = SpatialDropout2D(dropout)(h)
    h = MaxPooling2D(pool_size=(2, 2))(h)
    h = LeakyReLU(0.2)(h)
    h = Convolution2D(nch / 2, k, k, border_mode='same', W_regularizer=reg())(h)
    h = SpatialDropout2D(dropout)(h)
    h = MaxPooling2D(pool_size=(2, 2))(h)
    h = LeakyReLU(0.2)(h)
    h = Convolution2D(nch / 2, k, k, border_mode='same', W_regularizer=reg())(h)
    h = SpatialDropout2D(dropout)(h)
    h = MaxPooling2D(pool_size=(2, 2))(h)
    h = LeakyReLU(0.2)(h)
    h = Convolution2D(nch, k, k, border_mode='same', W_regularizer=reg())(h)
    h = SpatialDropout2D(dropout)(h)
    h = LeakyReLU(0.2)(h)
    h = Flatten()(h)
    mu = Dense(latent_dim, name="encoder_mu", W_regularizer=reg())(h)
    log_sigma_sq = Dense(latent_dim, name="encoder_log_sigma_sq", W_regularizer=reg())(h)
    z = merge([mu, log_sigma_sq], mode=lambda p: p[0] + K.random_normal(K.shape(p[0])) * K.exp(p[1] / 2),
              output_shape=lambda p: p[0])
    return Model(x, z, name="encoder")


def model_discriminator(latent_dim, output_dim=1, hidden_dim=256, reg=lambda: l1l2(1e-7, 1e-7)):
    z = Input((latent_dim,))
    h = z
    mode = 1
    h = Dense(hidden_dim, name="discriminator_h1", W_regularizer=reg())(h)
    h = BatchNormalization(mode=mode)(h)
    h = LeakyReLU(0.2)(h)
    h = Dense(hidden_dim, name="discriminator_h2", W_regularizer=reg())(h)
    h = BatchNormalization(mode=mode)(h)
    h = LeakyReLU(0.2)(h)
    h = Dense(hidden_dim, name="discriminator_h3", W_regularizer=reg())(h)
    h = BatchNormalization(mode=mode)(h)
    h = LeakyReLU(0.2)(h)
    y = Dense(output_dim, name="discriminator_y", activation="sigmoid", W_regularizer=reg())(h)
    return Model(z, y)


def example_aae(path, adversarial_optimizer):
    # z \in R^100
    latent_dim = 100
    # x \in R^{28x28}
    input_shape = dim_ordering_shape((3, 32, 32))

    # generator (z -> x)
    generator = model_generator(latent_dim)
    # encoder (x ->z)
    encoder = model_encoder(latent_dim, input_shape)
    # autoencoder (x -> x')
    autoencoder = Model(encoder.inputs, generator(encoder(encoder.inputs)))
    # discriminator (z -> y)
    discriminator = model_discriminator(latent_dim)

    # build AAE
    x = encoder.inputs[0]
    z = encoder(x)
    xpred = generator(z)
    zreal = normal_latent_sampling((latent_dim,))(x)
    yreal = discriminator(zreal)
    yfake = discriminator(z)
    aae = Model(x, fix_names([xpred, yfake, yreal], ["xpred", "yfake", "yreal"]))

    # print summary of models
    generator.summary()
    encoder.summary()
    discriminator.summary()
    autoencoder.summary()

    # build adversarial model
    generative_params = generator.trainable_weights + encoder.trainable_weights
    model = AdversarialModel(base_model=aae,
                             player_params=[generative_params, discriminator.trainable_weights],
                             player_names=["generator", "discriminator"])
    model.adversarial_compile(adversarial_optimizer=adversarial_optimizer,
                              player_optimizers=[Adam(1e-4, decay=1e-4), Adam(1e-3, decay=1e-4)],
                              loss={"yfake": "binary_crossentropy", "yreal": "binary_crossentropy",
                                    "xpred": "mean_squared_error"},
                              compile_kwargs={"loss_weights": {"yfake": 1e-2, "yreal": 1e-2, "xpred": 1}})

    # load mnist data
    xtrain, xtest = cifar10_data()

    # callback for image grid of generated samples
    def generator_sampler():
        zsamples = np.random.normal(size=(10 * 10, latent_dim))
        return dim_ordering_unfix(generator.predict(zsamples)).transpose((0, 2, 3, 1)).reshape((10, 10, 32, 32, 3))

    generator_cb = ImageGridCallback(os.path.join(path, "generated-epoch-{:03d}.png"), generator_sampler)

    # callback for image grid of autoencoded samples
    def autoencoder_sampler():
        xsamples = n_choice(xtest, 10)
        xrep = np.repeat(xsamples, 9, axis=0)
        xgen = dim_ordering_unfix(autoencoder.predict(xrep)).reshape((10, 9, 3, 32, 32))
        xsamples = dim_ordering_unfix(xsamples).reshape((10, 1, 3, 32, 32))
        samples = np.concatenate((xsamples, xgen), axis=1)
        samples = samples.transpose((0, 1, 3, 4, 2))
        print("Samples: {}".format(samples.shape))
        return samples

    autoencoder_cb = ImageGridCallback(os.path.join(path, "autoencoded-epoch-{:03d}.png"), autoencoder_sampler,
                                       cmap=None)

    # train network
    # generator, discriminator; pred, yfake, yreal
    n = xtrain.shape[0]
    y = [xtrain, np.ones((n, 1)), np.zeros((n, 1)), xtrain, np.zeros((n, 1)), np.ones((n, 1))]
    ntest = xtest.shape[0]
    ytest = [xtest, np.ones((ntest, 1)), np.zeros((ntest, 1)), xtest, np.zeros((ntest, 1)), np.ones((ntest, 1))]
    history = model.fit(x=xtrain, y=y, validation_data=(xtest, ytest),
                        callbacks=[generator_cb, autoencoder_cb],
                        nb_epoch=100, batch_size=32)

    # save history
    df = pd.DataFrame(history.history)
    df.to_csv(os.path.join(path, "history.csv"))

    # save model
    encoder.save(os.path.join(path, "encoder.h5"))
    generator.save(os.path.join(path, "generator.h5"))
    discriminator.save(os.path.join(path, "discriminator.h5"))


def main():
    example_aae("output/aae-cifar10", AdversarialOptimizerSimultaneous())


if __name__ == "__main__":
    main()
