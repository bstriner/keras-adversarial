from setuptools import setup, find_packages

version = '0.0.1'

setup(name='keras-adversarial',
      version=version,
      description='Adversarial models and optimizers for Keras',
      url='https://github.com/bstriner/keras_adversarial',
      author='bstriner',
      author_email='bstriner@gmail.com',
      packages=find_packages(),
      install_requires=[
          'keras>=1.1.2'
      ]
      )
