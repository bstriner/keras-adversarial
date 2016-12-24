import matplotlib as mpl

# This line allows mpl to run with no DISPLAY defined
mpl.use('Agg')
from example_gan import example_gan
from keras_adversarial.unrolled_optimizer import UnrolledAdversarialOptimizer
from keras.optimizers import Adam, SGD
from example_gan import model_generator, model_discriminator
from keras.regularizers import l1l2
import os

def example_gan_unrolled(path, depth_g, depth_d, clipvalue=2.0):
    # z \in R^100
    latent_dim = 100
    # x \in R^{28x28}
    input_shape = (28, 28)
    # generator (z -> x)
    generator = model_generator(latent_dim, input_shape, hidden_dim=512, batch_norm_mode=-1)
    # discriminator (x -> y)
    discriminator = model_discriminator(input_shape, hidden_dim=512, dropout=0, batch_norm_mode=-1)
    example_gan(UnrolledAdversarialOptimizer(depth_g=depth_g, depth_d=depth_d), path,
                opt_g=Adam(1e-4, decay=1e-4, clipvalue=clipvalue),#, nesterov=True, momentum=0.1),
                opt_d=Adam(1e-3, decay=1e-4, clipvalue=clipvalue),# nesterov=True, momentum=0.1),
                nb_epoch=50, generator=generator, discriminator=discriminator,
                latent_dim=latent_dim)


if __name__ == "__main__":

    path = "output/unrolled_gan"
    def example(name, depth_g, depth_d, clipvalue=2.0):
        example_gan_unrolled(os.path.join(path, name), depth_g, depth_d, clipvalue)
    example("k_0_0", 0, 0, 2)
    example("k_8_8_clip_0", 8, 8, 0)
    example("k_16_16_clip_0", 16, 16, 0)
    example("k_8_8_clip_2", 8, 8, 2)
    example("k_8_8_clip_0.1", 8, 8, 0.1)
    example("k_8_8_clip_0.5", 8, 8, 0.5)
    example("k_16_16_clip_2", 16, 16, 2)
    example("k_16_16_clip_0.5", 16, 16, 0.5)
    example("k_16_16_clip_10", 16, 16, 10)
    example("k_32_32", 32, 32)
    example("k_1_1", 1, 1)
    example("k_2_0", 2, 0)
    example("k_4_0", 4, 0)
    example("k_8_0", 8, 0)
