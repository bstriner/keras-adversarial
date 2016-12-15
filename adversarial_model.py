from keras import backend as K
from keras.models import Model, Sequential
from keras import optimizers, objectives
from keras.layers import Activation
import theano
import numpy as np


def build_gan(generator, discriminator):
    yfake = Activation("linear", name="yfake")(discriminator(generator(generator.inputs)))
    yreal = Activation("linear", name="yreal")(discriminator(discriminator.inputs))
    model = Model(generator.inputs + discriminator.inputs, [yfake, yreal])
    return model


def gan_targets(n):
    generator_fake = np.ones((n, 1))
    generator_real = np.zeros((n, 1))
    discriminator_fake = np.zeros((n, 1))
    discriminator_real = np.ones((n, 1))
    return [generator_fake, generator_real, discriminator_fake, discriminator_real]


def normal_latent_sampler(n, latent_shape):
    print "Sampler: %s, %s" % (str(n), str(latent_shape))
    return K.random_normal((n, ) + latent_shape)


class AdversarialModel(Model):
    def __init__(self, gan, generator_params, discriminator_params, latent_sampler=normal_latent_sampler):
        # assert (len(generator.inputs) == 1)
        # assert (len(discriminator.outputs) == 1)
        # assert (len(generator.outputs) == len(discriminator.inputs))

        self.gan = gan
        self.generator_params = generator_params
        self.discriminator_params = discriminator_params
        self.latent_sampler = latent_sampler
        self.generator_optimizer = None
        self.discriminator_optimizer = None
        self.loss = None
        self.optimizer = None
        self._function_kwargs = None
        self.latent_shape = self.gan.internal_input_shapes[0][1:]

    def adversarial_compile(self, adversarial_optimizer, generator_optimizer, discriminator_optimizer, loss,
                            **kwargs):
        '''Configures the learning process.'''
        self._function_kwargs = kwargs
        self.adversarial_optimizer = adversarial_optimizer
        self.generator_optimizer = optimizers.get(generator_optimizer)
        self.discriminator_optimizer = optimizers.get(discriminator_optimizer)
        self.loss = objectives.get(loss)
        self.optimizer = None

        # Build two GAN models
        self.gan_generator = Model(self.gan.inputs, self.gan(self.gan.inputs))
        self.gan_generator.compile(self.generator_optimizer, loss=self.loss)
        self.gan_discriminator = Model(self.gan.inputs, self.gan(self.gan.inputs))
        self.gan_discriminator.compile(self.discriminator_optimizer, loss=self.loss)

        self.layers = [self.gan_generator, self.gan_discriminator]
        self.train_function = None
        self.test_function = None

        # concatenate generator and discriminator
        self.internal_output_shapes = self.gan_generator.internal_output_shapes + \
                                      self.gan_discriminator.internal_output_shapes
        self.output_names = self.gan_generator.output_names + self.gan_discriminator.output_names
        self.loss_functions = self.gan_generator.loss_functions + self.gan_discriminator.loss_functions
        self.internal_input_shapes = self.gan_generator.internal_input_shapes[1]
        self.input_names = [self.gan_generator.input_names[1]]
        self.sample_weight_modes = self.gan_generator.sample_weight_modes + self.gan_discriminator.sample_weight_modes
        self.metrics_names = ["loss", "generator_loss", "discriminator_loss"]

        self.inputs = [self.gan_generator.inputs[1]]
        self.targets = self.gan_generator.targets + self.gan_discriminator.targets
        self.sample_weights = self.gan_generator.sample_weights + self.gan_discriminator.sample_weights

        latent_samples = self.latent_sampler(self.gan_generator.inputs[1].shape[0], self.latent_shape)

        generator_loss = self.gan_generator.total_loss
        generator_loss = theano.clone(generator_loss,
                                      replace={self.gan_generator.inputs[0]: latent_samples})
        self.generator_loss = generator_loss
        discriminator_loss = self.gan_discriminator.total_loss
        discriminator_loss = theano.clone(discriminator_loss,
                                          replace={self.gan_discriminator.inputs[0]: latent_samples})
        self.discriminator_loss = discriminator_loss
        self.total_loss = generator_loss + discriminator_loss

    @property
    def uses_learning_phase(self):
        return True

    @property
    def constraints(self):
        cons = self.gan_generator.constraints.copy()
        cons.update(self.gan_discriminator.constraints)
        return cons

    @property
    def regularizers(self):
        return self.gan_generator.regularizers + self.gan_discriminator.regularizers

    def _make_train_function(self):
        if not hasattr(self, 'train_function'):
            raise Exception('You must compile your model before using it.')
        if self.train_function is None:
            updates = self.adversarial_optimizer.call(self.generator_loss, self.discriminator_loss,
                                                      self.generator_params, self.discriminator_params,
                                                      self.generator_optimizer, self.discriminator_optimizer,
                                                      self.constraints)

            inputs = self.inputs + self.targets + self.sample_weights + [K.learning_phase()]

            # returns loss and metrics. Updates weights at each call.
            self.train_function = K.function(inputs,
                                             [self.total_loss, self.generator_loss, self.discriminator_loss],
                                             updates=updates,
                                             **self._function_kwargs)

    def _make_test_function(self):
        if not hasattr(self, 'test_function'):
            raise Exception('You must compile your model before using it.')
        if self.test_function is None:
            inputs = self.inputs + self.targets + self.sample_weights + [K.learning_phase()]

            # return loss and metrics, no gradient updates.
            # Does update the network states.
            self.test_function = K.function(inputs,
                                            [self.total_loss, self.generator_loss, self.discriminator_loss],
                                            # + self.metrics_tensors,
                                            updates=self.state_updates,
                                            **self._function_kwargs)
