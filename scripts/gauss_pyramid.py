# %%
import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.transform import pyramid_gaussian


def set_style():
    """set the style of the plots"""
    plt.style.use('fivethirtyeight')
    plt.rcParams['figure.figsize'] = [9.0, 5.0]
    plt.rcParams['figure.dpi'] = 240
    plt.rcParams['axes.linewidth'] = 1.25
    plt.rcParams['savefig.pad_inches'] = 0.2
    plt.rcParams['savefig.bbox'] = 'tight'


set_style()

# %%

image = data.astronaut()
rows, cols, dim = image.shape
pyramid = tuple(pyramid_gaussian(image, downscale=2, multichannel=True))

composite_image = np.zeros((rows, cols + cols // 2, dim), dtype=np.double)

composite_image[:rows, :cols, :] = pyramid[0]

i_row = 0
for p in pyramid[1:]:
    n_rows, n_cols = p.shape[:2]
    composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
    i_row += n_rows

fig, ax = plt.subplots()
ax.imshow(composite_image)
ax.set_title('Gaussian Pyramid')
ax.grid(False)
ax.set_xlabel('pixels')
ax.set_ylabel('pixels')
fig.savefig('gauss_pyramid.png')

# %%
