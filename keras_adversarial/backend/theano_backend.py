from theano import clone


def unpack_assignment(a):
    return a


def map_params(params):
    return params


def clone_replace(f, replace):
    return clone(f, replace=replace)

def variable_key(a):
    return a
