from keras_adversarial.adversarial_optimizers import AdversarialOptimizerSimultaneous
import keras.backend as K

if K.backend() == "tensorflow":
    import tensorflow as tf


    def f_replace(f, replace):
        replacements = {k.op.outputs[0]: v.op.outputs[0] for k, v in replace.iteritems()}
        return tf.contrib.graph_editor.graph_replace(f, replacements)
else:
    import theano


    def f_replace(f, replace):
        return theano.clone(f, replace=replace)


def unroll(loss, opt, params, constraint, uloss, uopt, uparams, uconstraint, depth):
    replace = {k: v for k, v in uopt.get_updates(uparams, uconstraint, uloss)}
    loss_t = loss
    for i in range(depth):
        loss_t = f_replace(loss_t, replace)
    updates = opt.get_updates(params, constraint, loss_t)
    return updates


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

        gupdates = unroll(losses[0], optimizers[0], params[0], constraints[0],
                          losses[1], optimizers[1], params[1], constraints[1],
                          self.depth_g)
        dupdates = unroll(losses[1], optimizers[1], params[1], constraints[1],
                          losses[0], optimizers[0], params[0], constraints[0],
                          self.depth_d)

        updates = gupdates + dupdates
        return updates
