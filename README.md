# Keras Adversarial Models

`AdversarialModel` simulates multi-player games. `AdversarialModel` takes as input a base model and a list of players.

The adversarial model has the same inputs as the base model and separate targets for each player.

If there are `n` players and the base model has `m` targets and `k` metrics:
* the adversarial model has `n*m` targets
* the adversarial model has `1+n*k` metrics

`AdversarialModel` is a subclass of `Model`, so can be trained like a normal Keras model.

`AdversarialOptimizer` is a base class that creates the training function:
* `AdversarialOptimizerSimultaneous` updates each player simultaneously
* `AdversarialOptimizerAlternating` updates each player in a round-robin

## Examples

### Example Generative Adversarial Network (GAN)

[example_gan.py](https://github.com/bstriner/keras_adversarial/blob/master/examples/example_gan.py) shows how to
create a GAN in Keras.

![Example GAN](https://github.com/bstriner/keras_adversarial/raw/master/doc/images/gan-epoch-099.png)

### Example Bi-Directional Generative Adversarial Network (BiGAN)

[example_bigan.py](https://github.com/bstriner/keras_adversarial/blob/master/examples/example_bigan.py) shows how to
 create a BiGAN in Keras.
