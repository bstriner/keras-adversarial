from keras.callbacks import Callback
from .image_grid import write_image_grid


class ImageGridCallback(Callback):
    def __init__(self, image_path, generator, cmap='gray'):
        self.image_path = image_path
        self.generator = generator
        self.cmap = cmap

    def on_epoch_end(self, epoch, logs={}):
        xsamples = self.generator()
        image_path = self.image_path.format(epoch)
        write_image_grid(image_path, xsamples, cmap=self.cmap)
