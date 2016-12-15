import matplotlib as mpl

mpl.use('Agg')
import os

os.environ["THEANO_FLAGS"] = "mode=FAST_COMPILE,device=cpu,floatX=float32"

import keras.backend as K
from keras.layers import Input, Dense
from keras.models import Model
from adversarial_model import AdversarialModel
from keras.datasets import mnist
import numpy as np


def model_generator(latent_dim, input_dim, hidden_dim=256, activation="tanh"):
    z = Input((latent_dim,), name="generator_z")
    h1 = Dense(hidden_dim, name="generator_h1", activation=activation)(z)
    h2 = Dense(hidden_dim, name="generator_h2", activation=activation)(h1)
    x = Dense(input_dim, name="generator_x", activation="sigmoid")(h2)
    model = Model(z, x, name="generator")
    return model


def model_discriminator(input_dim, hidden_dim=256, activation="tanh"):
    x = Input((input_dim,), name="discriminator_x")
    h1 = Dense(hidden_dim, name="discriminator_h1", activation=activation)(x)
    h2 = Dense(hidden_dim, name="discriminator_h2", activation=activation)(h1)
    y = Dense(input_dim, name="discriminator_y", activation="sigmoid")(h2)
    model = Model(x, y, name="discriminator")
    return model


def mnist_process(x):
    x = x.astype(np.float32) / 255.0
    x = x.reshape((-1, 28 * 28))
    return x


def mnist_data():
    (xtrain, ytrain), (xtest, ytest) = mnist.load_data()
    return mnist_process(xtrain), mnist_process(xtest)


if __name__ == "__main__":
    latent_dim = 100
    input_dim = 28 * 28
    generator = model_generator(latent_dim, input_dim)
    discriminator = model_discriminator(input_dim)

    print generator.inputs[0]
    print generator.inputs[0].shape
    print generator.internal_input_shapes

    gan = AdversarialModel(generator=generator, discriminator=discriminator)
    gan.adversarial_compile('adam', 'adam', 'binary_crossentropy')

    xtrain, xtest = mnist_data()
    n = xtrain.shape[0]
    # ytrain = np.ones((n,1))
    gan.fit(x=xtrain, y=[])
