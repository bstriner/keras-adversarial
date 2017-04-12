import numpy as np
import pytest
from keras.layers import LeakyReLU, Activation
from keras.models import Sequential
from keras.optimizers import Adam
from keras_adversarial import AdversarialModel, simple_gan, gan_targets
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling
from keras_adversarial.legacy import fit, Dense


def model_generator(latent_dim, input_dim, hidden_dim=256):
    return Sequential([
        Dense(hidden_dim, name="generator_h1", input_dim=latent_dim),
        LeakyReLU(0.2),
        Dense(hidden_dim, name="generator_h2"),
        LeakyReLU(0.2),
        Dense(hidden_dim, name="generator_h3"),
        LeakyReLU(0.2),
        Dense(input_dim, name="generator_x_flat")],
        name="generator")


def model_discriminator(input_dim, hidden_dim=256):
    return Sequential([
        Dense(hidden_dim, name="discriminator_h1", input_dim=input_dim),
        LeakyReLU(0.2),
        Dense(hidden_dim, name="discriminator_h2"),
        LeakyReLU(0.2),
        Dense(hidden_dim, name="discriminator_h3"),
        LeakyReLU(0.2),
        Dense(1, name="discriminator_y"),
        Activation('sigmoid')],
        name="discriminator")


def gan_model_test():
    latent_dim = 10
    input_dim = 5
    generator = model_generator(input_dim=input_dim, latent_dim=latent_dim)
    discriminator = model_discriminator(input_dim=input_dim)
    gan = simple_gan(generator, discriminator, normal_latent_sampling((latent_dim,)))

    # build adversarial model
    model = AdversarialModel(base_model=gan,
                             player_params=[generator.trainable_weights, discriminator.trainable_weights],
                             player_names=["generator", "discriminator"])
    adversarial_optimizer = AdversarialOptimizerSimultaneous()
    opt_g = Adam(1e-4)
    opt_d = Adam(1e-3)
    loss = 'binary_crossentropy'
    model.adversarial_compile(adversarial_optimizer=adversarial_optimizer,
                              player_optimizers=[opt_g, opt_d],
                              loss=loss)

    # train model
    batch_size = 32
    n = batch_size * 8
    x = np.random.random((n, input_dim))
    y = gan_targets(n)
    fit(model, x, y, nb_epoch=3, batch_size=batch_size)


if __name__ == "__main__":
    pytest.main([__file__])
