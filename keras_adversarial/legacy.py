"""
Utility functions to avoid warnings while testing both Keras 1 and 2.
"""
import keras

keras_2 = int(keras.__version__.split(".")[0]) > 1  # Keras > 1


def fit_generator(model, generator, epochs, steps_per_epoch):
    if keras_2:
        model.fit_generator(generator, epochs=epochs, steps_per_epoch=steps_per_epoch)
    else:
        model.fit_generator(generator, nb_epoch=epochs, samples_per_epoch=steps_per_epoch)


def fit(model, x, y, epochs, **kwargs):
    if keras_2:
        model.fit(x, y, epochs=epochs, **kwargs)
    else:
        model.fit(x, y, nb_epoch=epochs, **kwargs)
