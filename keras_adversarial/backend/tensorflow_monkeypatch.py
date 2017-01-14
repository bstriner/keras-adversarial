import keras.backend

"""
Import this file to monkeypatch tensorflow to lazily convert tuples to tf_assign inside K.function.
Makes unpacking and inspecting updates in tensorflow much cleaner.
"""


def update(x, new_x):
    return (x, new_x)


def update_add(x, increment):
    return (x, x + increment)


def update_sub(x, decrement):
    return (x, x - decrement)


def moving_average_update(variable, value, momentum):
    return (variable, variable * momentum + value * (1. - momentum))

keras.backend.update=update
keras.backend.update_add=update_add
keras.backend.update_sub=update_sub
keras.backend.moving_average_update=moving_average_update
