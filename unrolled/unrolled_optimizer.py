from adversarial.adversarial_optimizers import AdversarialOptimizerSimultaneous
import theano.tensor as T
import theano

class UnrolledAdversarialOptimizer(AdversarialOptimizerSimultaneous):
    def __init__(self, depth):
        self.depth = depth

    def call(self, losses, params, optimizers, constraints):
        # Players should be [generator, discriminator]
        assert (len(optimizers) == 2)

        # Unroll discriminator
        discriminator_params = params[1]
        discriminator_loss = losses[1]
        discriminator_optimizer = optimizers[1]
        discriminator_constraint = constraints[1]
        discriminator_updates = discriminator_optimizer.get_updates(discriminator_params, discriminator_constraint,
                                                                    discriminator_loss)
        discriminator_replacements = {k: v for k, v in discriminator_updates}
        updates_t = [(k, k) for k in discriminator_params]
        for i in range(self.depth):
            updates_t = [(k, theano.clone(v, replace=discriminator_replacements)) for k, v in updates_t]
        discriminator_replacements_t = {k: v for k,v in updates_t}

        generator_params = params[0]
        generator_loss = losses[0]
        generator_optimizer = optimizers[0]
        generator_constraint = constraints[0]
        generator_updates = generator_optimizer.get_updates(generator_params, generator_constraint, generator_loss)

        unrolled_generator_updates = [(k, theano.clone(v, replace=discriminator_replacements_t)) for k, v in generator_updates]

        updates = unrolled_generator_updates + discriminator_updates
        return updates
