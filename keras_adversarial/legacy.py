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


def l1l2(l1=0, l2=0):
    if keras_2:
        return keras.regularizers.L1L2(l1, l2)
    else:
        return keras.regularizers.l1l2(l1, l2)


def Dense(units, W_regularizer=None, **kwargs):
    if keras_2:
        return keras.layers.Dense(units, kernel_regularizer=W_regularizer, **kwargs)
    else:
        return keras.layers.Dense(units, W_regularizer=W_regularizer, **kwargs)


def BatchNormalization(mode=0, **kwargs):
    if keras_2:
        return keras.layers.BatchNormalization(**kwargs)
    else:
        return keras.layers.BatchNormalization(mode=mode, **kwargs)


def fit(model, nb_epoch=10, *args, **kwargs):
    if keras_2:
        return model.fit(*args, epochs=nb_epoch, **kwargs)
    else:
        return model.fit(*args, nb_epoch=nb_epoch, **kwargs)


def Convolution2D(units, w, h, W_regularizer=None, border_mode='same', **kwargs):
    if keras_2:
        return keras.layers.Convolution2D(units, (w, h), padding=border_mode, kernel_regularizer=W_regularizer,
                                          **kwargs)
    else:
        return keras.layers.Convolution2D(units, w, h, border_mode=border_mode, W_regularizer=W_regularizer, **kwargs)

def AveragePooling2D(pool_size, border_mode='valid', **kwargs):
    if keras_2:
        keras.layers.AveragePooling2D(pool_size=pool_size, padding=border_mode, **kwargs)
    else:
        keras.layers.AveragePooling2D(pool_size=pool_size, border_mode=border_mode, **kwargs)