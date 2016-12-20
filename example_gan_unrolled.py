import matplotlib as mpl

# This line allows mpl to run with no DISPLAY defined
mpl.use('Agg')
from example_gan import example_gan
from unrolled import UnrolledAdversarialOptimizer
from keras.optimizers import Adam

if __name__ == "__main__":
    nb_epoch = 20
    example_gan(UnrolledAdversarialOptimizer(depth=0), "output/unrolled_gan/k_0",
                Adam(1e-4, decay=1e-4),
                Adam(1e-3, decay=1e-4),
                nb_epoch)
    example_gan(UnrolledAdversarialOptimizer(depth=1), "output/unrolled_gan/k_1",
                Adam(1e-4, decay=1e-4),
                Adam(1e-3, decay=1e-4),
                nb_epoch)
    example_gan(UnrolledAdversarialOptimizer(depth=2), "output/unrolled_gan/k_2",
                Adam(1e-4, decay=1e-4),
                Adam(1e-3, decay=1e-4),
                nb_epoch)
    example_gan(UnrolledAdversarialOptimizer(depth=10), "output/unrolled_gan/k_10",
                Adam(1e-4, decay=1e-4),
                Adam(1e-3, decay=1e-4),
                nb_epoch)
