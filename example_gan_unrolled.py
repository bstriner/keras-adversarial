import matplotlib as mpl

# This line allows mpl to run with no DISPLAY defined
mpl.use('Agg')
from example_gan import example_gan
from keras_adversarial.unrolled_optimizer import UnrolledAdversarialOptimizer
from keras.optimizers import Adam
from example_gan import model_generator, model_discriminator

def example_gan_unrolled(path, depth):
    # z \in R^100
    latent_dim = 100
    # x \in R^{28x28}
    input_shape = (28, 28)
    # generator (z -> x)
    generator = model_generator(latent_dim, input_shape)
    # discriminator (x -> y)
    discriminator = model_discriminator(input_shape)
    example_gan(UnrolledAdversarialOptimizer(depth=depth), path,
                opt_g=Adam(1e-4, decay=1e-4, clipvalue=2.0),
                opt_d=Adam(1e-3, decay=1e-4, clipvalue=2.0),
                nb_epoch=50, generator=generator, discriminator=discriminator,
                latent_dim=latent_dim)


if __name__ == "__main__":
    example_gan_unrolled("output/unrolled_gan/k_0", 0)
    example_gan_unrolled("output/unrolled_gan/k_1", 1)
    example_gan_unrolled("output/unrolled_gan/k_2", 2)
    example_gan_unrolled("output/unrolled_gan/k_4", 4)
    example_gan_unrolled("output/unrolled_gan/k_8", 8)
