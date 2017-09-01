from abc import ABCMeta, abstractmethod

import keras.backend as K

from .legacy import get_updates


class AdversarialOptimizer(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def make_train_function(self, inputs, outputs, losses, params, optimizers, constraints, model_updates,
                            function_kwargs):
        """
        Construct function that updates weights and returns losses.
        :param inputs: function inputs
        :param outputs: function outputs
        :param losses: player losses
        :param params: player parameters
        :param optimizers: player optimizers
        :param constraints: player constraints
        :param function_kwargs: function kwargs
        :return:
        """
        pass


class AdversarialOptimizerSimultaneous(object):
    """
    Perform simultaneous updates for each player in the game.
    """

    def make_train_function(self, inputs, outputs, losses, params, optimizers, constraints, model_updates,
                            function_kwargs):
        return K.function(inputs,
                          outputs,
                          updates=self.call(losses, params, optimizers, constraints) + model_updates,
                          **function_kwargs)

    def call(self, losses, params, optimizers, constraints):
        updates = []
        for loss, param, optimizer, constraint in zip(losses, params, optimizers, constraints):
            updates += optimizer.get_updates(param, constraint, loss)
        return updates


class AdversarialOptimizerAlternating(object):
    """
    Perform round-robin updates for each player in the game. Each player takes a turn.
    Take each batch and run that batch through each of the models. All models are trained on each batch.
    """

    def __init__(self, reverse=False):
        """
        Initialize optimizer.
        :param reverse: players take turns in reverse order
        """
        self.reverse = reverse

    def make_train_function(self, inputs, outputs, losses, params, optimizers, constraints, model_updates,
                            function_kwargs):
        funcs = []
        for loss, param, optimizer, constraint in zip(losses, params, optimizers, constraints):
            updates = optimizer.get_updates(param, constraint, loss)
            funcs.append(K.function(inputs, [], updates=updates, **function_kwargs))
        output_func = K.function(inputs, outputs, updates=model_updates, **function_kwargs)
        if self.reverse:
            funcs = funcs.reverse()

        def train(_inputs):
            # update each player
            for func in funcs:
                func(_inputs)
            # return output
            return output_func(_inputs)

        return train


class AdversarialOptimizerScheduled(object):
    """
    Perform updates according to a schedule.
    For example, [0,0,1] will train player 0 on batches 0,1,3,4,6,7... and player 1 on batches 2,5,8...
    """

    def __init__(self, schedule):
        """
        Initialize optimizer.
        :param schedule: Schedule of updates
        """
        assert len(schedule) > 0
        self.schedule = schedule
        self.iter = 0

    def make_train_function(self, inputs, outputs, losses, params, optimizers, constraints, model_updates,
                            function_kwargs):
        funcs = []
        for loss, param, optimizer, constraint in zip(losses, params, optimizers, constraints):
            updates = get_updates(optimizer=optimizer, params=param, constraints=constraint, loss=loss)
            funcs.append(K.function(inputs, outputs, updates=updates + model_updates, **function_kwargs))

        def train(_inputs):
            self.iter += 1
            if self.iter == len(self.schedule):
                self.iter = 0
            func = funcs[self.schedule[self.iter]]
            return func(_inputs)

        return train
