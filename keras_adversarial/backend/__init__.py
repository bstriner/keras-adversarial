import keras.backend as K


if K.backend() == "tensorflow":
    from .tensorflow_backend import unpack_assignment, clone_replace, map_params, variable_key
else:
    from .theano_backend import unpack_assignment, clone_replace, map_params, variable_key

def unpack_assignments(assignments):
    return [unpack_assignment(a) for a in assignments]
