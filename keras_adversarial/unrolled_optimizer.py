from keras_adversarial.adversarial_optimizers import AdversarialOptimizerSimultaneous
import keras.backend as K

if K.backend() == "tensorflow":
    import tensorflow as tf
    from tensorflow.contrib.graph_editor import select
    from six import iterkeys
    from tensorflow.contrib.graph_editor import util
    from tensorflow.python.framework import ops as tf_ops

    def unpack_assignment(a):
        return a.op.inputs[0], a.op.inputs[1]


    def map_params(params):
        return [x.op.outputs[0] for x in params]


    def f_replace(f, replace):
        flatten_target_ts = util.flatten_tree(f)
        graph = util.get_unique_graph(flatten_target_ts, check_types=(tf_ops.Tensor))
        control_ios = util.ControlOutputs(graph)
        ops = select.get_walks_intersection_ops(list(iterkeys(replace)),
                                                flatten_target_ts,
                                                control_ios=control_ios)
        if not ops:
            #print "Disconnected: {}".format(f)
            return f
        else:
            return tf.contrib.graph_editor.graph_replace(f, replace)
else:
    import theano


    def unpack_assignment(a):
        return a


    def map_params(params):
        return params


    def f_replace(f, replace):
        return theano.clone(f, replace=replace)


def unroll(updates, uupdates, depth):
    replace = {unpack_assignment(a)[0]: unpack_assignment(a)[1] for a in uupdates}
    #print "Replacements: {}".format(len(replace))
    updates_t = [unpack_assignment(a) for a in updates]
    for i in range(depth):
        updates_t = [(k, f_replace(v, replace)) for k, v in updates_t]
    return [K.update(a,b) for a,b in updates_t]


class UnrolledAdversarialOptimizer(AdversarialOptimizerSimultaneous):
    def __init__(self, depth_g, depth_d):
        """
        :param depth_g: Depth to unroll discriminator when updating generator
        :param depth_d: Depth to unroll generator when updating discriminator
        """
        self.depth_g = depth_g
        self.depth_d = depth_d

    def call(self, losses, params, optimizers, constraints):
        # Players should be [generator, discriminator]
        assert (len(optimizers) == 2)

        updates = [o.get_updates(p, c, l) for o, p, c, l in zip(optimizers, params, constraints, losses)]

        gupdates = unroll(updates[0], updates[1], self.depth_g)
        dupdates = unroll(updates[1], updates[0], self.depth_d)

        return gupdates + dupdates
