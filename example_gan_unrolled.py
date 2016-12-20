import matplotlib as mpl

# This line allows mpl to run with no DISPLAY defined
mpl.use('Agg')
from example_gan import example_gan
from unrolled import UnrolledAdversarialOptimizer

if __name__ == "__main__":
    example_gan(UnrolledAdversarialOptimizer(depth=2), "output/unrolled-gan")
