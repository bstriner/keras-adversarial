import matplotlib as mpl

# This line allows mpl to run with no DISPLAY defined
mpl.use('Agg')

import pandas as pd
import numpy as np
import os
from keras.layers import Dense, Reshape, Flatten, Dropout, LeakyReLU, Activation, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l1, l1l2
from keras.datasets import mnist
from keras_adversarial import AdversarialModel, ImageGridCallback, simple_gan, gan_targets
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling


def model_generator(latent_dim, input_shape, hidden_dim=1024, reg=lambda: l1(1e-5), batch_norm_mode=1):
    return Sequential([
        Dense(hidden_dim / 4, name="generator_h1", input_dim=latent_dim, W_regularizer=reg()),
        BatchNormalization(mode=batch_norm_mode),
        #Activation('relu'),
        LeakyReLU(0.2),
        Dense(hidden_dim / 2, name="generator_h2", W_regularizer=reg()),
        BatchNormalization(mode=batch_norm_mode),
        #Activation('relu'),
        LeakyReLU(0.2),
        Dense(hidden_dim, name="generator_h3", W_regularizer=reg()),
        BatchNormalization(mode=batch_norm_mode),
        #Activation('relu'),
        LeakyReLU(0.2),
        Dense(np.prod(input_shape), name="generator_x_flat", W_regularizer=reg()),
        Activation('sigmoid'),
        Reshape(input_shape, name="generator_x")],
        name="generator")


def model_discriminator(input_shape, hidden_dim=1024, reg=lambda: l1l2(1e-5, 1e-5), dropout=0.5, batch_norm_mode=1):
    return Sequential([
        Flatten(name="discriminator_flatten", input_shape=input_shape),
        Dense(hidden_dim, name="discriminator_h1", W_regularizer=reg()),
        BatchNormalization(mode=batch_norm_mode),
        LeakyReLU(0.2),
        Dropout(dropout),
        Dense(hidden_dim / 2, name="discriminator_h2", W_regularizer=reg()),
        BatchNormalization(mode=batch_norm_mode),
        LeakyReLU(0.2),
        Dropout(dropout),
        Dense(hidden_dim / 4, name="discriminator_h3", W_regularizer=reg()),
        BatchNormalization(mode=batch_norm_mode),
        LeakyReLU(0.2),
        Dropout(dropout),
        Dense(1, name="discriminator_y", W_regularizer=reg()),
        Activation("sigmoid")],
        name="discriminator")


def mnist_process(x):
    x = x.astype(np.float32) / 255.0
    return x


def mnist_data():
    (xtrain, ytrain), (xtest, ytest) = mnist.load_data()
    return mnist_process(xtrain), mnist_process(xtest)

def example_gan(adversarial_optimizer, path, opt_g, opt_d, nb_epoch, generator, discriminator, latent_dim):

    # gan (x - > yfake, yreal), z generated on GPU
    gan = simple_gan(generator, discriminator, normal_latent_sampling((latent_dim,)))

    # print summary of models
    generator.summary()
    discriminator.summary()
    gan.summary()

    # build adversarial model
    model = AdversarialModel(base_model=gan,
                             player_params=[generator.trainable_weights, discriminator.trainable_weights],
                             player_names=["generator", "discriminator"])
    model.adversarial_compile(adversarial_optimizer=adversarial_optimizer,
                              player_optimizers=[opt_g, opt_d],
                              loss='binary_crossentropy')

    # train model
    def generator_sampler():
        zsamples = np.random.normal(size=(10 * 10, latent_dim))
        return generator.predict(zsamples).reshape((10, 10, 28, 28))

    generator_cb = ImageGridCallback(os.path.join(path, "epoch-{:03d}.png"), generator_sampler)

    xtrain, xtest = mnist_data()
    y = gan_targets(xtrain.shape[0])
    ytest = gan_targets(xtest.shape[0])
    history = model.fit(x=xtrain, y=y, validation_data=(xtest, ytest), callbacks=[generator_cb], nb_epoch=nb_epoch,
                        batch_size=32)
    df = pd.DataFrame(history.history)
    df.to_csv(os.path.join(path, "history.csv"))

    generator.save(os.path.join(path, "generator.h5"))
    discriminator.save(os.path.join(path, "discriminator.h5"))


if __name__ == "__main__":
    # z \in R^100
    latent_dim = 100
    # x \in R^{28x28}
    input_shape = (28, 28)
    # generator (z -> x)
    generator = model_generator(latent_dim, input_shape)
    # discriminator (x -> y)
    discriminator = model_discriminator(input_shape)
    example_gan(AdversarialOptimizerSimultaneous(), "output/gan",
                opt_g=Adam(1e-4, decay=1e-4, clipvalue=2.0),
                opt_d=Adam(1e-3, decay=1e-4, clipvalue=2.0),
                nb_epoch=100, generator=generator, discriminator=discriminator,
                latent_dim=latent_dim)
