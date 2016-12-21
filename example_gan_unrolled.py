import matplotlib as mpl

# This line allows mpl to run with no DISPLAY defined
mpl.use('Agg')
from example_gan import example_gan
from unrolled import UnrolledAdversarialOptimizer
from keras.optimizers import SGD


def example_gan_unrolled(path, depth):
    example_gan(UnrolledAdversarialOptimizer(depth=depth), path,
                SGD(3e-4, decay=1e-4),
                SGD(1e-4, decay=1e-4),
                nb_epoch=50)


if __name__ == "__main__":
    example_gan_unrolled("output/unrolled_gan/k_0", 0)
    example_gan_unrolled("output/unrolled_gan/k_1", 1)
    example_gan_unrolled("output/unrolled_gan/k_2", 2)
    example_gan_unrolled("output/unrolled_gan/k_4", 4)
    example_gan_unrolled("output/unrolled_gan/k_8", 8)
