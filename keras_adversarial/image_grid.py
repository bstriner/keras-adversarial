from matplotlib import pyplot as plt, gridspec

import os


def write_image_grid(filepath, imgs, figsize=None, cmap='gray'):
    directory = os.path.dirname(os.path.abspath(filepath))
    if not os.path.exists(directory):
        os.makedirs(directory)
    fig = create_image_grid(imgs, figsize, cmap=cmap)
    fig.savefig(filepath)
    plt.close(fig)


def create_image_grid(imgs, figsize=None, cmap='gray'):
    n = imgs.shape[0]
    m = imgs.shape[1]
    if figsize is None:
        figsize = (n, m)
    fig = plt.figure(figsize=figsize)
    gs1 = gridspec.GridSpec(n, m)
    gs1.update(wspace=0.025, hspace=0.025)  # set the spacing between axes.
    for i in range(n):
        for j in range(m):
            ax = plt.subplot(gs1[i, j])
            img = imgs[i, j, :]
            ax.imshow(img, cmap=cmap)
            ax.axis('off')
    return fig
