# %%

import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.transform import rescale
from skimage.color import rgb2gray
from skimage.feature import corner_harris, corner_peaks


def set_style():
    """set the style of the plots"""
    plt.style.use('fivethirtyeight')
    plt.rcParams['figure.figsize'] = [9.0, 5.0]
    plt.rcParams['figure.dpi'] = 240
    plt.rcParams['axes.linewidth'] = 1.25
    plt.rcParams['savefig.pad_inches'] = 0.2
    plt.rcParams['savefig.bbox'] = 'tight'


set_style()


# %% rescale(image, 0.25, anti_aliasing=False)

astro = data.astronaut()
nd1 = rescale(plt.imread("data/notre_dame_01.png")[:, :, :3],
              [0.25, 0.25, 1], anti_aliasing=True)
nd2 = rescale(plt.imread("data/notre_dame_02.png")[:, :, :3],
              [0.25, 0.25, 1], anti_aliasing=True)


nd1_gray = rgb2gray(nd1)
nd2_gray = rgb2gray(nd2)

# %%

# takes a while to run...
nd1_corners = corner_peaks(corner_harris(
    nd1_gray), min_distance=4, threshold_rel=0.0001)
nd2_corners = corner_peaks(corner_harris(
    nd2_gray), min_distance=4, threshold_rel=0.0001)


y, x = np.transpose(nd1_corners)
fig, ax, = plt.subplots()
ax.imshow(nd1_gray, cmap='gray')
ax.plot(x, y, 'r.', markersize=3)
ax.axis('off')
