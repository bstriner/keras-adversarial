import tensorflow as tf
from tensorflow.contrib.graph_editor import select
from six import iterkeys
from tensorflow.contrib.graph_editor import util
from tensorflow.python.framework import ops as tf_ops


def unpack_assignment(a):
    if isinstance(a, (list, tuple)):
        assert (len(a) == 2)
        return a
    elif isinstance(a, tf.Tensor):
        assert (a.op.type in ['Assign', 'AssignAdd', 'AssignSub'])
        if a.op.type == 'Assign':
            return a.op.inputs[0], a.op.inputs[1]
        if a.op.type == 'AssignAdd':
            return a.op.inputs[0], a.op.inputs[0] + a.op.inputs[1]
        elif a.op.type == 'AssignSub':
            return a.op.inputs[0], a.op.inputs[0] - a.op.inputs[1]
        else:
            raise ValueError("Unsupported operation: {}".format(a.op.type))
    else:
        raise ValueError("Unsupported assignment object type: {}".format(type(a)))


def map_params(params):
    return [x.op.outputs[0] for x in params]


def clone_replace(f, replace):
    flatten_target_ts = util.flatten_tree(f)
    graph = util.get_unique_graph(flatten_target_ts, check_types=(tf_ops.Tensor))
    control_ios = util.ControlOutputs(graph)
    ops = select.get_walks_intersection_ops(list(iterkeys(replace)),
                                            flatten_target_ts,
                                            control_ios=control_ios)
    if not ops:
        # this happens with disconnected inputs
        return f
    else:
        return tf.contrib.graph_editor.graph_replace(f, replace)


def variable_key(a):
    if hasattr(a, "op"):
        return a.op
    else:
        return a
