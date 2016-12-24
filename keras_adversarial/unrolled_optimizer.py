from keras_adversarial.adversarial_optimizers import AdversarialOptimizerSimultaneous
import keras.backend as K

if K.backend() == "tensorflow":
    import tensorflow as tf


    def unpack_assignment(a):
        return a.op.inputs[0], a.op.inputs[1]


    def map_params(params):
        return [x.op.outputs[0] for x in params]


    def f_replace(f, replace):
        return tf.contrib.graph_editor.graph_replace(f, replace)
else:
    import theano


    def unpack_assignment(a):
        return a


    def map_params(params):
        return params


    def f_replace(f, replace):
        return theano.clone(f, replace=replace)


def unroll(loss, opt, params, constraint, uloss, uopt, uparams, uconstraint, depth, params_only=False):
    replace = {unpack_assignment(a)[0]: unpack_assignment(a)[1] for a in uopt.get_updates(uparams, uconstraint, uloss)}
    if params_only:
        replace = {k: v for k, v in replace.iteritems() if k in map_params(uparams)}
    print "Replacements: {}".format(len(replace))
    loss_t = loss
    for i in range(depth):
        loss_t = f_replace(loss_t, replace)
    updates = opt.get_updates(params, constraint, loss_t)
    return updates


class UnrolledAdversarialOptimizer(AdversarialOptimizerSimultaneous):
    def __init__(self, depth_g, depth_d, params_only):
        """
        :param depth_g: Depth to unroll discriminator when updating generator
        :param depth_d: Depth to unroll generator when updating discriminator
        """
        self.depth_g = depth_g
        self.depth_d = depth_d
        self.params_only = params_only

    def call(self, losses, params, optimizers, constraints):
        # Players should be [generator, discriminator]
        assert (len(optimizers) == 2)

        gupdates = unroll(losses[0], optimizers[0], params[0], constraints[0],
                          losses[1], optimizers[1], params[1], constraints[1],
                          self.depth_g, self.params_only)
        dupdates = unroll(losses[1], optimizers[1], params[1], constraints[1],
                          losses[0], optimizers[0], params[0], constraints[0],
                          self.depth_d, self.params_only)

        updates = gupdates + dupdates
        return updates
