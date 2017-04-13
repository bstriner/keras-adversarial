Keras Adversarial Models
========================

**Combine multiple models into a single Keras model. GANs made easy!**

``AdversarialModel`` simulates multi-player games. A single call to
``model.fit`` takes targets for each player and updates all of the
players. Use ``AdversarialOptimizer`` for complete control of whether
updates are simultaneous, alternating, or something else entirely. No
more fooling with ``Trainable`` either!

Installation
------------

.. code:: shell

    git clone https://github.com/bstriner/keras_adversarial.git
    cd keras_adversarial
    python setup.py install

Usage
-----

Please check the examples folder for exemplary usage.

Instantiating an adversarial model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Build separate models for each component / player such as generator
   and discriminator.
-  Build a combined model. For a GAN, this might have an input for
   images and an input for noise and an output for D(fake) and an output
   for D(real)
-  Pass the combined model and the separate models to the
   ``AdversarialModel`` constructor

.. code:: python

    adversarial_model = AdversarialModel(base_model=gan,
      player_params=[generator.trainable_weights, discriminator.trainable_weights],
      player_names=["generator", "discriminator"])

The resulting model will have the same inputs as ``gan`` but separate
targets and metrics for each player. This is accomplished by copying the
model for each player. If each player has a different model, use
``player_models`` (see below regarding dropout).

.. code:: python

    adversarial_model = AdversarialModel(player_models=[gan_g, gan_d],
      player_params=[generator.trainable_weights, discriminator.trainable_weights],
      player_names=["generator", "discriminator"])

Compiling an adversarial model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``adversarial_compile`` to compile the model. The parameters are an
``AdversarialOptimizer`` and a list of ``Optimizer`` objects for each
player. The loss is passed to ``model.compile`` for each model, so may
be a dictionary or other object. Use the same order for
``player_optimizers`` as you did for ``player_params`` and
``player_names``.

.. code:: python

    model.adversarial_compile(adversarial_optimizer=adversarial_optimizer,
      player_optimizers=[Adam(1e-4, decay=1e-4), Adam(1e-3, decay=1e-4)],
      loss='binary_crossentropy')

Training a simple adversarial model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adversarial models can be trained using ``fit`` and callbacks just like
any other Keras model. Just make sure to provide the correct targets in
the correct order.

For example, given simple GAN named ``gan``:

- Inputs: ``[x]``
- Targets: ``[y_fake, y_real]``
- Metrics: ``[loss, loss_y_fake, loss_y_real]``

``AdversarialModel(base_model=gan, player_names=['g', 'd']...)`` will have:

- Inputs: ``[x]``
- Targets: ``[g_y_fake, g_y_real, d_y_fake, d_y_real]``
- Metrics: ``[loss, g_loss, g_loss_y_fake, g_loss_y_real, d_loss, d_loss_y_fake, d_loss_y_real]``

Adversarial Optimizers
----------------------

There are many possible strategies for optimizing multiplayer games.
``AdversarialOptimizer`` is a base class that abstracts those strategies
and is responsible for creating the training function.

- ``AdversarialOptimizerSimultaneous`` updates each player simultaneously on each batch.
- ``AdversarialOptimizerAlternating`` updates each player in a round-robin.
  Take each batch and run that batch through each of the models. All models are trained on each batch.
- ``AdversarialOptimizerScheduled`` passes each batch to a different player according to a schedule.
  ``[1,1,0]`` would mean train player 1 on batches 0,1,3,4,6,7,etc. and train player 0 on batches 2,5,8,etc.
- ``UnrolledAdversarialOptimizer`` unrolls updates to stabilize training
  (only tested in Theano; slow to build graph but runs reasonably fast)

Examples
--------

MNIST Generative Adversarial Network (GAN)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`example\_gan.py <https://github.com/bstriner/keras_adversarial/blob/master/examples/example_gan.py>`__
shows how to create a GAN in Keras for the MNIST dataset.

.. figure:: https://github.com/bstriner/keras_adversarial/raw/master/doc/images/gan-epoch-099.png
   :alt: Example GAN

   Example GAN

CIFAR10 Generative Adversarial Network (GAN)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`example\_gan\_cifar10.py <https://github.com/bstriner/keras_adversarial/blob/master/examples/example_gan_cifar10.py>`__
shows how to create a GAN in Keras for the CIFAR10 dataset.

.. figure:: https://github.com/bstriner/keras_adversarial/raw/master/doc/images/gan-cifar10-epoch-099.png
   :alt: Example GAN

   Example GAN

MNIST Bi-Directional Generative Adversarial Network (BiGAN)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`example\_bigan.py <https://github.com/bstriner/keras_adversarial/blob/master/examples/example_bigan.py>`__
shows how to create a BiGAN in Keras.

.. figure:: https://github.com/bstriner/keras_adversarial/raw/master/doc/images/bigan-epoch-099.png
   :alt: Example BiGAN

   Example BiGAN

MNIST Adversarial Autoencoder (AAE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An AAE is like a cross between a GAN and a Variational Autoencoder
(VAE).
`example\_aae.py <https://github.com/bstriner/keras_adversarial/blob/master/examples/example_aae.py>`__
shows how to create an AAE in Keras.

.. figure:: https://github.com/bstriner/keras_adversarial/raw/master/doc/images/aae-epoch-099.png
   :alt: Example AAE

   Example AAE

Unrolled Generative Adversarial Network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`example\_gan\_unrolled.py <https://github.com/bstriner/keras_adversarial/blob/master/examples/example_gan_unrolled.py>`__
shows how to use the unrolled optimizer.

WARNING: Unrolling the discriminator 8 times takes about 6 hours to
build the function on my computer, but only a few minutes for epoch of
training. Be prepared to let it run a long time or turn the depth down
to around 4.

Notes
-----

Dropout
~~~~~~~

When training adversarial models using dropout, you may want to create
separate models for each player.

If you want to train a discriminator with dropout, but train the
generator against the discriminator without dropout, create two models.
\* GAN to train generator: ``D(G(z, dropout=0.5), dropout=0)`` \* GAN to
train discriminator: ``D(G(z, dropout=0), dropout=0.5)``

If you create separate models, use ``player_models`` parameter of
``AdversarialModel`` constructor.

If you aren't using dropout, one model is sufficient, and use
``base_model`` parameter of ``AdversarialModel`` constructor, which will
duplicate the ``base_model`` for each player.

Theano and Tensorflow
~~~~~~~~~~~~~~~~~~~~~

I do most of my development in theano but try to test tensorflow when I
have extra time. The goal is to support both. Please let me know any
issues you have with either backend.

Questions?
~~~~~~~~~~

Feel free to start an issue or a PR here or in Keras if you are having
any issues or think of something that might be useful.
