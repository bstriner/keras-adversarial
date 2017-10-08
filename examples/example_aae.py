# import os
# os.environ["THEANO_FLAGS"] = "mode=FAST_COMPILE,device=cpu,floatX=float32"

import matplotlib as mpl

# This line allows mpl to run with no DISPLAY defined
mpl.use('Agg')

from keras.layers import Dense, Reshape, Flatten, Input, merge
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras_adversarial.legacy import l1l2
import keras.backend as K
import pandas as pd
import numpy as np
from keras_adversarial.image_grid_callback import ImageGridCallback

from keras_adversarial import AdversarialModel, fix_names, n_choice
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling
from mnist_utils import mnist_data
from keras.layers import LeakyReLU, Activation
import os


def model_generator(latent_dim, input_shape, hidden_dim=512, reg=lambda: l1l2(1e-7, 0)):
    return Sequential([
        Dense(hidden_dim, name="generator_h1", input_dim=latent_dim, W_regularizer=reg()),
        LeakyReLU(0.2),
        Dense(hidden_dim, name="generator_h2", W_regularizer=reg()),
        LeakyReLU(0.2),
        Dense(np.prod(input_shape), name="generator_x_flat", W_regularizer=reg()),
        Activation('sigmoid'),
        Reshape(input_shape, name="generator_x")],
        name="generator")


def model_encoder(latent_dim, input_shape, hidden_dim=512, reg=lambda: l1l2(1e-7, 0)):
    x = Input(input_shape, name="x")
    h = Flatten()(x)
    h = Dense(hidden_dim, name="encoder_h1", W_regularizer=reg())(h)
    h = LeakyReLU(0.2)(h)
    h = Dense(hidden_dim, name="encoder_h2", W_regularizer=reg())(h)
    h = LeakyReLU(0.2)(h)
    mu = Dense(latent_dim, name="encoder_mu", W_regularizer=reg())(h)
    log_sigma_sq = Dense(latent_dim, name="encoder_log_sigma_sq", W_regularizer=reg())(h)
    z = merge([mu, log_sigma_sq], mode=lambda p: p[0] + K.random_normal(K.shape(p[0])) * K.exp(p[1] / 2),
              output_shape=lambda p: p[0])
    return Model(x, z, name="encoder")


def model_discriminator(latent_dim, output_dim=1, hidden_dim=512,
                        reg=lambda: l1l2(1e-7, 1e-7)):
    z = Input((latent_dim,))
    h = z
    h = Dense(hidden_dim, name="discriminator_h1", W_regularizer=reg())(h)
    h = LeakyReLU(0.2)(h)
    h = Dense(hidden_dim, name="discriminator_h2", W_regularizer=reg())(h)
    h = LeakyReLU(0.2)(h)
    y = Dense(output_dim, name="discriminator_y", activation="sigmoid", W_regularizer=reg())(h)
    return Model(z, y)


def example_aae(path, adversarial_optimizer):
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
    # discriminator (z -> y)
    discriminator = model_discriminator(latent_dim)

    # assemple AAE
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
                              player_compile_kwargs=[{"loss_weights": {"yfake": 1e-2, "yreal": 1e-2, "xpred": 1}}] * 2)

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
        samples = np.concatenate((xsamples, xgen), axis=1)
        return samples

    autoencoder_cb = ImageGridCallback(os.path.join(path, "autoencoded-epoch-{:03d}.png"), autoencoder_sampler)

    # train network
    # generator, discriminator; pred, yfake, yreal
    n = xtrain.shape[0]
    y = [xtrain, np.ones((n, 1)), np.zeros((n, 1)), xtrain, np.zeros((n, 1)), np.ones((n, 1))]
    ntest = xtest.shape[0]
    ytest = [xtest, np.ones((ntest, 1)), np.zeros((ntest, 1)), xtest, np.zeros((ntest, 1)), np.ones((ntest, 1))]
    history = model.fit(x=xtrain, y=y, validation_data=(xtest, ytest), callbacks=[generator_cb, autoencoder_cb],
                        nb_epoch=100, batch_size=32)

    # save history
    df = pd.DataFrame(history.history)
    df.to_csv(os.path.join(path, "history.csv"))

    # save model
    encoder.save(os.path.join(path, "encoder.h5"))
    generator.save(os.path.join(path, "generator.h5"))
    discriminator.save(os.path.join(path, "discriminator.h5"))


def main():
    example_aae("output/aae", AdversarialOptimizerSimultaneous())


if __name__ == "__main__":
    main()
