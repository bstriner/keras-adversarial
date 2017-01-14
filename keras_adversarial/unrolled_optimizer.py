from .adversarial_optimizers import AdversarialOptimizerSimultaneous
from .backend import unpack_assignments, clone_replace
import keras.backend as K


def unroll(updates, uupdates, depth):
    replace = {k: v for k, v in unpack_assignments(uupdates)}
    updates_t = unpack_assignments(updates)
    for i in range(depth):
        updates_t = [(k, clone_replace(v, replace)) for k, v in updates_t]
    return [K.update(a, b) for a, b in updates_t]


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

        updates = [o.get_updates(p, c, l) for o, p, c, l in zip(optimizers, params, constraints, losses)]

        gupdates = unroll(updates[0], updates[1], self.depth_g)
        dupdates = unroll(updates[1], updates[0], self.depth_d)

        return gupdates + dupdates
