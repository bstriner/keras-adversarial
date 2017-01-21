from setuptools import setup, find_packages

long_description = open('README.rst').read()
version = '0.0.3'

setup(name='keras-adversarial',
      version=version,
      description='Adversarial models and optimizers for Keras',
      url='https://github.com/bstriner/keras-adversarial',
      download_url='https://github.com/bstriner/keras-adversarial/tarball/v{}'.format(version),
      author='Ben Striner',
      author_email='bstriner@gmail.com',
      packages=find_packages(),
      install_requires=['Keras'],
      keywords=['keras', 'gan', 'adversarial', 'multiplayer'],
      license='MIT',
      long_description=long_description,
      classifiers=[
          # Indicate who your project is intended for
          'Intended Audience :: Developers',
          # Pick your license as you wish (should match "license" above)
          'License :: OSI Approved :: MIT License',

          # Specify the Python versions you support here. In particular, ensure
          # that you indicate whether you support Python 2, Python 3 or both.
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 3'
      ])
