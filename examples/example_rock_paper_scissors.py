import os

os.environ["THEANO_FLAGS"] = "mode=FAST_COMPILE,device=cpu,floatX=float32"

"""Example of a two player game, rock paper scissors.

This game does not converge under simple alternating or simultaneous descent,
but converges using UnrolledAdversarialOptimizer.

"""
from keras_adversarial.adversarial_optimizers import AdversarialOptimizerSimultaneous, AdversarialOptimizerAlternating
from keras_adversarial.unrolled_optimizer import UnrolledAdversarialOptimizer
from keras_adversarial.adversarial_model import AdversarialModel
from keras.layers import Dense, merge, Input
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import LambdaCallback
from keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt
import os


def rps_chart(path, a, b):
    """Bar chart of two players in rock, paper, scissors"""
    fig, ax = plt.subplots()
    n = 3
    width = 0.35
    pad = 1.0 - 2 * width
    ind = np.arange(n)
    ba = plt.bar(pad/2 + ind, a, width=width, color='r')
    bb = plt.bar(pad/2 + ind + width, b, width=width, color='g')
    ax.set_ylabel('Frequency')
    ax.set_xticks(pad/2 + ind + width)
    ax.set_xticklabels(("Rock", "Paper", "Scissors"))
    fig.legend((ba, bb), ("Player A", "Player B"))
    ax.set_ylim([0, 1])
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    fig.savefig(path)
    plt.close(fig)


def experiment(opt, path):
    """Train two players to play rock, paper, scissors using a given optimizer"""
    x = Input((1,), name="x")
    player_a = Dense(3, activation='softmax', name="player_a", bias=False, W_regularizer=l2(1e-2))
    player_b = Dense(3, activation='softmax', name="player_b", bias=False, W_regularizer=l2(1e-2))

    action_a = player_a(x)
    action_b = player_b(x)

    def rps(z):
        u = z[0]
        v = z[1]
        return u[:, 0] * v[:, 2] + u[:, 1] * v[:, 0] + u[:, 2] * v[:, 1]

    model_a = Model(x, merge([action_a, action_b], mode=rps, output_shape=lambda z: (z[0][0], 1)))
    model_b = Model(x, merge([action_b, action_a], mode=rps, output_shape=lambda z: (z[0][0], 1)))

    adversarial_model = AdversarialModel(player_models=[model_a, model_b],
                                         player_params=[[player_a.W], [player_b.W]],
                                         player_names=["a", "b"])
    adversarial_model.adversarial_compile(opt,
                                          player_optimizers=[SGD(1), SGD(1)],
                                          loss="mean_absolute_error")
    param_model = Model(x, [action_a, action_b])

    def print_params(epoch, logs):
        params = param_model.predict(np.ones((1, 1)))
        a = params[0].ravel()
        b = params[1].ravel()
        print("Epoch: {}, A: {}, B: {}".format(epoch, a, b))
        imgpath = os.path.join(path, "epoch-{:03d}.png".format(epoch))
        rps_chart(imgpath, a, b)

    cb = LambdaCallback(on_epoch_begin=print_params)
    batch_count = 5
    adversarial_model.fit(np.ones((batch_count, 1)),
                          [np.ones((batch_count, 1)), np.ones((batch_count, 1))],
                          nb_epoch=120, callbacks=[cb], verbose=0, batch_size=1)


if __name__ == "__main__":
    experiment(AdversarialOptimizerSimultaneous(), "output/rock_paper_scissors/simultaneous")
    experiment(AdversarialOptimizerAlternating(), "output/rock_paper_scissors/alternating")
    experiment(UnrolledAdversarialOptimizer(depth_d=30, depth_g=30), "output/rock_paper_scissors/unrolled")
    experiment(UnrolledAdversarialOptimizer(depth_d=0, depth_g=30), "output/rock_paper_scissors/unrolled_player_a")
