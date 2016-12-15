import matplotlib as mpl

mpl.use('Agg')
# import os

# os.environ["THEANO_FLAGS"] = "mode=FAST_COMPILE,device=cpu,floatX=float32"

import keras.backend as K
from keras.layers import Input, Dense, Reshape, Flatten
from keras.models import Model
from adversarial_model import AdversarialModel, build_gan, gan_targets
from keras.datasets import mnist
import numpy as np
from adversarial_optimizers import AdversarialOptimizer, AdversarialOptimizerSimultaneous
from generate_samples_callback import GenerateSamplesCallback
from keras.optimizers import Adam


def model_generator(latent_dim, input_shape, hidden_dim=256, activation="tanh"):
    z = Input((latent_dim,), name="generator_z")
    h1 = Dense(hidden_dim, name="generator_h1", activation=activation)(z)
    h2 = Dense(hidden_dim, name="generator_h2", activation=activation)(h1)
    h3 = Dense(np.prod(input_shape), name="generator_h3", activation="sigmoid")(h2)
    output = Reshape(input_shape, name="generator_x")(h3)
    model = Model(z, output, name="generator")
    return model


def model_discriminator(input_shape, output_dim=1, hidden_dim=256, activation="tanh"):
    x = Input(input_shape, name="discriminator_x")
    flat = Flatten(name="discriminator_flatten")(x)
    h1 = Dense(hidden_dim, name="discriminator_h1", activation=activation)(flat)
    h2 = Dense(hidden_dim, name="discriminator_h2", activation=activation)(h1)
    y = Dense(output_dim, name="discriminator_y", activation="sigmoid")(h2)
    model = Model(x, y, name="discriminator")
    return model


def mnist_process(x):
    x = x.astype(np.float32) / 255.0
    #    x = x.reshape((-1, 28 * 28))
    return x


def mnist_data():
    (xtrain, ytrain), (xtest, ytest) = mnist.load_data()
    return mnist_process(xtrain), mnist_process(xtest)


if __name__ == "__main__":
    latent_dim = 100
    input_shape = (28, 28)
    generator = model_generator(latent_dim, input_shape)
    discriminator = model_discriminator(input_shape)
    gan = build_gan(generator, discriminator)

    model = AdversarialModel(gan=gan, generator_params=generator.trainable_weights,
                             discriminator_params=discriminator.trainable_weights)
    model.adversarial_compile(AdversarialOptimizerSimultaneous(),
                              generator_optimizer=Adam(1e-4, decay=1e-4),
                              discriminator_optimizer=Adam(1e-3, decay=1e-4),
                              loss='binary_crossentropy')

    zsamples = np.random.normal(size=(10 * 10, latent_dim))
    cb = GenerateSamplesCallback("samples/epoch-{}.png", lambda: generator.predict(zsamples), (10, 10))

    xtrain, xtest = mnist_data()
    n = xtrain.shape[0]
    ntest = xtest.shape[0]
    y = gan_targets(n)
    ytest = gan_targets(ntest)
    model.fit(x=xtrain, y=y, validation_data=(xtest, ytest), callbacks=[cb], nb_epoch=50, batch_size=32)
