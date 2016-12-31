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

### Generative Adversarial Network (GAN)

[example_gan.py](https://github.com/bstriner/keras_adversarial/blob/master/examples/example_gan.py) shows how to
create a GAN in Keras.

![Example GAN](https://github.com/bstriner/keras_adversarial/raw/master/doc/images/gan-epoch-099.png)

### Bi-Directional Generative Adversarial Network (BiGAN)

[example_bigan.py](https://github.com/bstriner/keras_adversarial/blob/master/examples/example_bigan.py) shows how to
 create a BiGAN in Keras.
  
![Example BiGAN](https://github.com/bstriner/keras_adversarial/raw/master/doc/images/bigan-epoch-099.png)

### Unrolled Generative Adversarial Network

[example_gan_unrolled.py](https://github.com/bstriner/keras_adversarial/blob/master/examples/example_gan_unrolled.py)
shows how to use the unrolled optimizer.

WARNING: Unrolling the discriminator 8 times takes about 6 hours to build the function on my computer,
but only a few minutes for epoch of training. Be prepared to let it run a long time or turn the depth down to around 4.

##Notes

###Dropout
When training adversarial models using dropout, you may want to create separate models for each player. 

If you want to train a discriminator with dropout, but train the generator against the discriminator without dropout, 
create two models.
* GAN to train generator: `D(G(z, dropout=0.5), dropout=0)`
* GAN to train discriminator: `D(G(z, dropout=0), dropout=0.5)`

If you create separate models, use `player_models` parameter of `AdversarialModel` constructor.

If you aren't using dropout, one model is sufficient, and use `base_model` parameter of `AdversarialModel` constructor,
 which will duplicate the `base_model` for each player.
 
###Theano and Tensorflow
I do most of my development in theano but try to test tensorflow when I have extra time. The goal is
to support both. Please let me know any issues you have with either backend.