import keras.backend as K


class AdversarialOptimizer(object):
    def call(self, generator_loss, discriminator_loss,
             generator_params, discriminator_params,
             generator_optimizer, discriminator_optimizer,
             constraints):
        return []

class AdversarialOptimizerSimultaneous(object):
    def call(self, generator_loss, discriminator_loss,
             generator_params, discriminator_params,
             generator_optimizer, discriminator_optimizer,
             constraints):
        generator_updates = generator_optimizer.get_updates(generator_params,
                                                            constraints,
                                                            generator_loss)
        discriminator_updates = discriminator_optimizer.get_updates(discriminator_params,
                                                                    constraints,
                                                                    discriminator_loss)
        updates = generator_updates + discriminator_updates
        return updates

class AdversarialOptimizerUnrolled(object):
    def call(self, generator_loss, discriminator_loss,
             generator_params, discriminator_params,
             generator_optimizer, discriminator_optimizer,
             constraints, unroll_depth):
        generator_updates = generator_optimizer.get_updates(generator_params,
                                                            constraints,
                                                            generator_loss)
        discriminator_updates = discriminator_optimizer.get_updates(discriminator_params,
                                                                    constraints,
                                                                    discriminator_loss)
        updates = generator_updates + discriminator_updates
        return updates