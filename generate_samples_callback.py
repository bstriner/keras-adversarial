from keras.callbacks import Callback
from image_grid import write_image_grid


class GenerateSamplesCallback(Callback):
    def __init__(self, image_path, generator, shape):
        self.image_path = image_path
        self.generator = generator
        self.shape = shape

    def on_epoch_end(self, epoch, logs={}):
        xsamples = self.generator()
        xsamples = xsamples.reshape(self.shape + xsamples.shape[1:])
        image_path = self.image_path.format(epoch)
        write_image_grid(image_path, xsamples)
