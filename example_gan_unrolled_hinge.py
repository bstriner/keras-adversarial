import matplotlib as mpl

# This line allows mpl to run with no DISPLAY defined
mpl.use('Agg')
from example_gan import example_gan
from keras_adversarial.unrolled_optimizer import UnrolledAdversarialOptimizer
from keras.optimizers import Adam
from example_gan import model_generator, model_discriminator
from keras_adversarial import gan_targets_hinge
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, LeakyReLU, BatchNormalization
from keras.regularizers import l1l2


def example_gan_unrolled_hinge(path, depth):
    # z \in R^100
    latent_dim = 100
    # x \in R^{28x28}
    input_shape = (28, 28)
    # generator (z -> x)
    generator = model_generator(latent_dim, input_shape, hidden_dim=512, batch_norm_mode=-1)
    # discriminator (x -> y)
    discriminator = model_discriminator(input_shape, output_activation='linear', hidden_dim=512, batch_norm_mode=-1,
                                        dropout=0)
    example_gan(UnrolledAdversarialOptimizer(depth=depth), path,
                opt_g=Adam(1e-4, decay=1e-4, clipvalue=2.0),
                opt_d=Adam(1e-3, decay=1e-4, clipvalue=2.0),
                nb_epoch=50, generator=generator, discriminator=discriminator,
                latent_dim=latent_dim, loss="squared_hinge", targets=gan_targets_hinge)


if __name__ == "__main__":
    example_gan_unrolled_hinge("output/unrolled_gan_hinge/k_0", 0)
    example_gan_unrolled_hinge("output/unrolled_gan_hinge/k_1", 1)
    example_gan_unrolled_hinge("output/unrolled_gan_hinge/k_2", 2)
    example_gan_unrolled_hinge("output/unrolled_gan_hinge/k_4", 4)
    example_gan_unrolled_hinge("output/unrolled_gan_hinge/k_8", 8)
