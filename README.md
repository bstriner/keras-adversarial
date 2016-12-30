# Keras Adversarial Models

AdversarialModel simulates multi-player games. AdversarialModel takes as input a base model and a list of players.

The adversarial model has the same inputs as the base model and separate targets for each player.

If there are n players and the base model has m targets and k metrics:
* the adversarial model has n*m targets
* the adversarial model has 1+n*k metrics

AdversarialModel is a subclass of Model, so can be trained like a normal Keras model.

AdversarialOptimizer is a base class that creates the training function:
* AdversarialOptimizerSimultaneous updates each player simultaneously
* AdversarialOptimizerAlternating updates each player in a round-robin

## Example GAN
example_gan contains an example of using AdversarialModel to train a GAN



example_bigan contains an example of using AdversarialModel to train a BiGAN
