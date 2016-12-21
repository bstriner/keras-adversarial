from keras import backend as K
from keras.models import Model
from keras import optimizers, objectives
import numpy as np
import itertools
from adversarial_utils import fix_names, merge_updates


class AdversarialModel(Model):
    """
    Adversarial training for multi-player games.
    Given a base model with n targets and k players, create a model with n*k targets.
    Each player optimizes loss on that player's targets.
    """

    def __init__(self, base_model, player_params, player_names=None):
        """
        Initialize adversarial model.
        :param base_model: base model will be duplicated for each player
        :param player_params: list of list of player parameters
        :param player_names: names of each player
        """

        self.base_model = base_model
        assert (len(player_params) > 0)
        self.player_params = player_params
        self.player_count = len(self.player_params)
        if player_names is None:
            player_names = ["player_{}".format(i) for i in range(self.player_count)]
        assert (len(player_names) == self.player_count)
        self.player_names = player_names

        self.generator_optimizer = None
        self.discriminator_optimizer = None
        self.loss = None
        self.total_loss = None
        self.optimizer = None
        self._function_kwargs = None

    def adversarial_compile(self, adversarial_optimizer, player_optimizers, loss,
                            **kwargs):
        """
        Configures the learning process.
        :param adversarial_optimizer: instance of AdversarialOptimizer
        :param player_optimizers: list of optimizers for each player
        :param loss: loss function or function name
        :param kwargs: additional arguments to function compilation
        :return:
        """
        self._function_kwargs = kwargs
        self.adversarial_optimizer = adversarial_optimizer
        assert (len(player_optimizers) == self.player_count)

        self.optimizers = [optimizers.get(optimizer) for optimizer in player_optimizers]
        self.loss = objectives.get(loss)
        self.optimizer = None

        # Build player models
        self.layers = []
        for i in range(self.player_count):
            # duplicate base model
            model = Model(self.base_model.inputs,
                          fix_names(self.base_model(self.base_model.inputs), self.base_model.output_names))
            # compile model
            model.compile(self.optimizers[i], loss=self.loss)
            # add model to list
            self.layers.append(model)

        self.train_function = None
        self.test_function = None

        # Inputs are same as base model
        self.internal_input_shapes = self.base_model.internal_input_shapes
        self.input_names = self.base_model.input_names
        self.inputs = self.base_model.inputs

        # Outputs are concatenated player models
        models = self.layers

        def collect(f):
            return list(itertools.chain.from_iterable(f(m) for m in models))

        self.internal_output_shapes = collect(lambda m: m.internal_output_shapes)
        self.loss_functions = collect(lambda m: m.loss_functions)
        self.targets = collect(lambda m: m.targets)
        self.outputs = collect(lambda m: m.outputs)
        self.sample_weights = collect(lambda m: m.sample_weights)
        self.sample_weight_modes = collect(lambda m: m.sample_weight_modes)
        # for each target, output name is {player}_{target}
        self.output_names = []
        for i in range(self.player_count):
            for name in models[i].output_names:
                self.output_names.append("{}_{}".format(self.player_names[i], name))
        # for each metric, metric name is {player}_{metric}
        self.metrics_names = ["loss"]
        for i in range(self.player_count):
            for name in models[i].metrics_names:
                self.metrics_names.append("{}_{}".format(self.player_names[i], name))

        # total loss is sum of losses
        self.total_loss = np.float32(0)
        for model in models:
            self.total_loss += model.total_loss

    @property
    def constraints(self):
        return list(itertools.chain.from_iterable(model.constraints for model in self.layers))

    @property
    def updates(self):
        #print "Updates: %s" % str(self.base_model.updates)
        #print "Updates: %s" % str(list(itertools.chain.from_iterable(model.updates for model in self.layers)))
        #print "Updates: %s" % str(merge_updates(self.base_model.updates))
        #return merge_updates(self.base_model.updates)
        return merge_updates(list(itertools.chain.from_iterable(model.updates for model in self.layers)))

    @property
    def regularizers(self):
        return list(itertools.chain.from_iterable(model.regularizers for model in self.layers))

    def _make_train_function(self):
        if not hasattr(self, 'train_function'):
            raise Exception('You must compile your model before using it.')
        if self.train_function is None:
            inputs = self.inputs + self.targets + self.sample_weights
            if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
                inputs += [K.learning_phase()]
            outputs = [self.total_loss] + \
                      list(itertools.chain.from_iterable(
                          [model.total_loss] + model.metrics_tensors
                          for model in self.layers))

            # returns loss and metrics. Updates weights at each call.
            self.train_function = self.adversarial_optimizer.make_train_function(inputs, outputs,
                                                                                 [model.total_loss for model in
                                                                                  self.layers],
                                                                                 self.player_params,
                                                                                 self.optimizers,
                                                                                 [model.constraints for model in
                                                                                  self.layers],
                                                                                 self.updates,
                                                                                 self._function_kwargs)

    def _make_test_function(self):
        if not hasattr(self, 'test_function'):
            raise Exception('You must compile your model before using it.')
        if self.test_function is None:
            inputs = self.inputs + self.targets + self.sample_weights
            if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
                inputs += [K.learning_phase()]
            outputs = [self.total_loss] + \
                      list(itertools.chain.from_iterable(
                          [model.total_loss] + model.metrics_tensors
                          for model in self.layers))
            self.test_function = K.function(inputs,
                                            outputs,
                                            updates=self.state_updates,
                                            **self._function_kwargs)
