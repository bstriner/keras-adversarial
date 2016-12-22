from keras_adversarial.adversarial_optimizers import AdversarialOptimizer

import keras.backend as K

if K.backend()=="tensorflow":
    import tensorflow as tf
    def f_replace(f, replace):
        return tf.contrib.graph_editor.copy_with_input_replacements(f, replace)
else:
    import theano
    def f_replace(f, replace):
        return theano.clone(f, replace=replace)


class UnrolledAdversarialOptimizer(AdversarialOptimizer):
    def __init__(self, depth):
        self.depth = depth

    def make_train_function(self, inputs, outputs, losses, params, optimizers, constraints, model_updates,
                            function_kwargs):
        return K.function(inputs,
                          outputs,
                          updates=self.call(losses, params, optimizers, constraints, model_updates) + model_updates,
                          **function_kwargs)

    def call(self, losses, params, optimizers, constraints, model_updates):
        # Players should be [generator, discriminator]
        assert (len(optimizers) == 2)

        # Unroll discriminator
        discriminator_params = params[1]
        discriminator_loss = losses[1]
        discriminator_optimizer = optimizers[1]
        discriminator_constraint = constraints[1]
        discriminator_updates = discriminator_optimizer.get_updates(discriminator_params, discriminator_constraint,
                                                                    discriminator_loss)
        updates_t = [(k[0], k[0]) for k in discriminator_updates + model_updates]
        discriminator_replacements = {k: v for k, v in discriminator_updates + model_updates}
        for i in range(self.depth):
            updates_t = [(k, f_replace(v, discriminator_replacements)) for k, v in updates_t]
        discriminator_replacements_t = {k: v for k, v in updates_t}

        generator_params = params[0]
        generator_loss = losses[0]
        generator_optimizer = optimizers[0]
        generator_constraint = constraints[0]
        generator_updates = generator_optimizer.get_updates(generator_params, generator_constraint, generator_loss)

        unrolled_generator_updates = [(k, f_replace(v, discriminator_replacements_t)) for k, v in
                                      generator_updates]

        updates = unrolled_generator_updates + discriminator_updates
        return updates
