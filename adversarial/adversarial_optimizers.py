import keras.backend as K


class AdversarialOptimizer(object):
    def call(self, losses, params, optimizers, constraints):
        return []


class AdversarialOptimizerSimultaneous(object):
    def call(self, losses, params, optimizers, constraints):
        updates = []
        for loss, param, optimizer, constraint in zip(losses, params, optimizers, constraints):
            updates += optimizer.get_updates(param, constraint, loss)
        return updates

