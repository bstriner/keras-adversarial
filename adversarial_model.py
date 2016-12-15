from keras import backend as K
from keras.models import Model, Sequential
from keras import optimizers, objectives
from keras.layers import Activation
import theano

def normal_latent_sampler(n, latent_dim):
    print "Sampler: %s, %s" % (str(n), str(latent_dim))
    return K.random_normal((n,latent_dim))

class AdversarialModel(Model):
    def __init__(self, generator, discriminator, latent_sampler=normal_latent_sampler):
        assert (len(generator.inputs) == 1)
        assert (len(discriminator.outputs) == 1)
        assert (len(generator.outputs) == len(discriminator.inputs))

        self.generator = generator
        self.discriminator = discriminator
        self.latent_sampler = latent_sampler
        self.generator_optimizer = None
        self.discriminator_optimizer = None
        self.loss = None
        self.optimizer = None
        self._function_kwargs = None
        self.latent_shape = self.generator.internal_input_shapes[0][1:]

    def adversarial_compile(self, generator_optimizer, discriminator_optimizer, loss,
                metrics=[],
                sample_weight_mode=None,
                target_real=1, target_fake=0,
                **kwargs):
        '''Configures the learning process.'''
        self._function_kwargs = kwargs
        self.generator_optimizer = optimizers.get(generator_optimizer)
        self.discriminator_optimizer = optimizers.get(discriminator_optimizer)
        self.loss = objectives.get(loss)
        self.optimizer = None
        # generator_params = self.generator.tr
        # print dir(self.generator)
        # print self.generator.trainable_weights
        # print self.discriminator.trainable_weights

        #latent_samples = self.latent_sampler(self.discriminator.inputs[0].shape[0], self.generator.inputs[0].shape[1])

        yfake = Activation('linear', name='yfake')(self.discriminator(self.generator(self.generator.inputs)))
        yreal = Activation('linear', name='yreal')(self.discriminator(self.discriminator.inputs))

        # build generator
        def generator_loss_real(ytrue, ypred):
            return self.loss(target_fake, ypred)

        def generator_loss_fake(ytrue, ypred):
            return self.loss(target_real, ypred)

        self.gan_generator = Model(self.generator.inputs + self.discriminator.inputs, [yfake, yreal])
        self.gan_generator.compile(self.generator_optimizer,
                                   loss={"yfake": generator_loss_fake, "yreal": generator_loss_real})

        # build discriminator
        def discriminator_loss_real(ytrue, ypred):
            return self.loss(target_real, ypred)

        def discriminator_loss_fake(ytrue, ypred):
            return self.loss(target_fake, ypred)

        self.gan_discriminator = Model(self.generator.inputs + self.discriminator.inputs, [yfake, yreal])
        self.gan_discriminator.compile(self.discriminator_optimizer,
                                       loss={"yfake": discriminator_loss_fake, "yreal": discriminator_loss_real})

        self.train_function = None
        self.internal_output_shapes = []
        self.output_names = []
        self.loss_functions = []
        self.internal_input_shapes = self.discriminator.internal_input_shapes
        self.input_names = self.discriminator.input_names
        self.sample_weight_modes = []
        self.metrics_names = ["loss", "generator_loss", "discriminator_loss"]

    def _standardize_user_data(self, x, y,
                               sample_weight=None, class_weight=None,
                               check_batch_dim=True, batch_size=None):
        # print "Sample weight: %s"%str(sample_weight)
        return [x], [], []

    @property
    def uses_learning_phase(self):
        return True

    @property
    def constraints(self):
        cons = self.generator.constraints.copy()
        cons.update(self.discriminator.constraints)
        return cons

    @property
    def regularizers(self):
        regs = self.generator.regularizers.copy()
        regs.update(self.discriminator.regularizers)
        return regs

    def _make_train_function(self):
        if not hasattr(self, 'train_function'):
            raise Exception('You must compile your model before using it.')
        if self.train_function is None:
            generator_loss = self.gan_generator.total_loss
            discriminator_loss = self.gan_discriminator.total_loss
            latent_input = self.gan_generator.inputs[0]
            print "input: %s"%str(latent_input)
            inputs = theano.gof.graph.inputs([generator_loss])
            #print "generator_loss inputs: %s"%str(inputs)

            latent_samples = self.latent_sampler(self.discriminator.inputs[0].shape[0], self.latent_shape[0])
            generator_loss = theano.clone(generator_loss, replace={latent_input: latent_samples})
            inputs = theano.gof.graph.inputs([generator_loss])
            print "generator_loss inputs: %s" % str(inputs)

            discriminator_loss = theano.clone(discriminator_loss, replace={latent_input: latent_samples})
            inputs = theano.gof.graph.inputs([discriminator_loss])
            print "discriminator_loss inputs: %s" % str(inputs)

            generator_updates = self.generator_optimizer.get_updates(self.generator.trainable_weights,
                                                                     self.constraints,
                                                                     generator_loss)
            discriminator_updates = self.discriminator_optimizer.get_updates(self.discriminator.trainable_weights,
                                                                             self.constraints,
                                                                             discriminator_loss)
            updates = generator_updates + discriminator_updates
            #inputs = self.generator.inputs + self.discriminator.inputs
            inputs = self.discriminator.inputs
            #sample_weights = self.gan_generator.sample_weights + self.gan_discriminator.sample_weights
            #targets = self.gan_generator.targets + self.gan_discriminator.targets

            #fun_inputs = inputs + targets + sample_weights + [K.learning_phase()]
            fun_inputs = inputs
            # returns loss and metrics. Updates weights at each call.
            self.total_loss = generator_loss + discriminator_loss
            self.train_function = K.function(fun_inputs,
                                             [self.total_loss, generator_loss,
                                              discriminator_loss],
                                             updates=updates,
                                             **self._function_kwargs)
