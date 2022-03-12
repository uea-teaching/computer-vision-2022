# %%

from matplotlib import tight_layout
import matplotlib.pyplot as plt

from skimage import data, io
from skimage.color import rgb2gray
from skimage.filters import difference_of_gaussians, gaussian


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

image = rgb2gray(data.astronaut())
filtered_image = difference_of_gaussians(image, 1.5)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(image, cmap='gray')
ax[0].title.set_text('Original')
ax[0].axis('off')
ax[1].imshow(filtered_image, cmap='gray')
ax[1].title.set_text('Difference of Gaussians')
ax[1].axis('off')

fig.savefig('dog-example.png')

# %%

img1 = rgb2gray(io.imread('data/norwich_cathedral.png')[:, :, :3])

# %%

fig, axs = plt.subplots(2, 4, tight_layout=True)
for i, ax in enumerate(axs.flat):
    if i == 0:
        im, title = img1, 'original'
    else:
        sigma = 2**(i-1)
        title = fr'$\sigma$ = {sigma}'
        im = gaussian(img1, sigma=sigma)
    ax.imshow(im, cmap='gray')
    ax.set_title(title, fontsize=12)
    ax.axis('off')

fig.savefig('scale-space.png')

# %%
