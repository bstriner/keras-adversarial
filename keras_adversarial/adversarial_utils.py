from keras.layers import Activation, Lambda
import numpy as np
import keras.backend as K
from keras.models import Model


def build_gan(generator, discriminator, name="gan"):
    """
    Build GAN from generator and discriminator
    Model is (z, x) -> (yfake, yreal)
    :param generator: Model (z -> x)
    :param discriminator: Model (x -> y)
    :return: GAN model
    """
    yfake = Activation("linear", name="yfake")(discriminator(generator(generator.inputs)))
    yreal = Activation("linear", name="yreal")(discriminator(discriminator.inputs))
    model = Model(generator.inputs + discriminator.inputs, [yfake, yreal], name=name)
    return model


def eliminate_z(gan, latent_sampling):
    """
    Eliminate z from GAN using latent_sampling
    :param gan: model with 2 inputs: z, x
    :param latent_sampling: layer that samples z with same batch size as x
    :return: Model x -> gan(latent_sampling(x), x)
    """
    x = gan.inputs[1]
    z = latent_sampling(x)
    model = Model(x, fix_names(gan([z, x]), gan.output_names), name=gan.name)
    return model


def simple_gan(generator, discriminator, latent_sampling):
    # build basic gan
    gan = build_gan(generator, discriminator)
    # generate z on gpu, eliminate one input
    gan = eliminate_z(gan, latent_sampling)
    return gan


def simple_bigan(generator, encoder, discriminator, latent_sampling):
    """
    Construct BiGRAN x -> yfake, yreal
    :param generator: model z->x
    :param encoder: model x->z
    :param discriminator: model z,x->y (z must be first)
    :param latent_sampling: layer for sampling from latent space
    :return:
    """
    zfake = latent_sampling(discriminator.inputs[1])
    xreal = discriminator.inputs[1]
    xfake = generator(zfake)
    zreal = encoder(xreal)
    yfake = discriminator([zfake, xfake])
    yreal = discriminator([zreal, xreal])
    bigan = Model(xreal, fix_names([yfake, yreal], ["yfake", "yreal"]), name="bigan")
    return bigan


def fix_names(outputs, names):
    if not isinstance(outputs, list):
        outputs = [outputs]
    if not isinstance(names, list):
        names = [names]
    return [Activation('linear', name=name)(output) for output, name in zip(outputs, names)]


def gan_targets(n):
    """
    Standard training targets
    [generator_fake, generator_real, discriminator_fake, discriminator_real] = [1, 0, 0, 1]
    :param n: number of samples
    :return: array of targets
    """
    generator_fake = np.ones((n, 1))
    generator_real = np.zeros((n, 1))
    discriminator_fake = np.zeros((n, 1))
    discriminator_real = np.ones((n, 1))
    return [generator_fake, generator_real, discriminator_fake, discriminator_real]

def gan_targets_hinge(n):
    """
    Standard training targets for hinge loss
    [generator_fake, generator_real, discriminator_fake, discriminator_real] = [1, -1, -1, 1]
    :param n: number of samples
    :return: array of targets
    """
    generator_fake = np.ones((n, 1))
    generator_real = np.ones((n, 1))*-1
    discriminator_fake = np.ones((n, 1))*-1
    discriminator_real = np.ones((n, 1))
    return [generator_fake, generator_real, discriminator_fake, discriminator_real]


def normal_latent_sampling(latent_shape):
    """
    Sample from normal distribution
    :param latent_shape: batch shape
    :return: normal samples, shape=(n,)+latent_shape
    """
    return Lambda(lambda x: K.random_normal((K.shape(x)[0],) + latent_shape),
                  output_shape=lambda x: ((x[0],) + latent_shape))


def uniform_latent_sampling(latent_shape, low=0.0, high=1.0):
    """
    Sample from uniform distribution
    :param latent_shape: batch shape
    :return: normal samples, shape=(n,)+latent_shape
    """
    return Lambda(lambda x: K.random_uniform((K.shape(x)[0],) + latent_shape, low, high),
                  output_shape=lambda x: ((x[0],) + latent_shape))


def n_choice(x, n):
    return x[np.random.choice(x.shape[0], size=n, replace=False)]

def merge_updates(updates):
    """Average repeated updates of the same variable"""
    upd = {}
    for k,v in updates:
        if k not in upd:
            upd[k] = []
        upd[k].append(v)
    ret = []
    for k,v in upd.iteritems():
        n = len(v)
        if n==1:
            ret.append((k, v[0]))
        else:
            ret.append((k, sum(v)/n))
    return ret

