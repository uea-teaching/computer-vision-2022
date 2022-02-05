# %%
import json
import numpy as np
from sklearn.decomposition import PCA
from utils import procrustes
import matplotlib.pyplot as plt
from pathlib import Path
plt.style.use('fivethirtyeight')

# %%

# Load data and centre it
with open('lmk_recG001_crop.json') as fid:
    mouths = np.array(json.load(fid), dtype=np.float64)[:, 48:, :]
    mouths -= np.mean(mouths, axis=1, keepdims=True)

# %%

mean_mouth = np.mean(mouths, axis=0, keepdims=True)
mean_mouth -= mean_mouth.mean(axis=1, keepdims=True)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.set_title('Mean mouth')
ax.scatter(mean_mouth[0, :, 0], mean_mouth[0, :, 1])
ax.scatter(mouths[0, :, 0], mouths[0, :, 1])
ax.legend(['mean', 'first'])

# %%

d, Z, r = procrustes(mean_mouth[0], mouths[0], scaling=False)

print(d)
print(Z)
print(r)

# %%

# Align all the mouths to the mean mouth
aligned = np.stack([procrustes(mean_mouth[0], m, scaling=False)[1]
                   for m in mouths], 0)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.set_title('aligned mouths')
for a in aligned[:50]:
    ax.scatter(a[:, 0], a[:, 1])

# %%

idx = 42
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.set_title('Procrustes')
ax.scatter(aligned[idx, :, 0], aligned[idx, :, 1])
ax.legend(['mean', 'first'])


# %%

# Perform PCA
x_mean = mean_mouth.reshape(1, 40)
x_aligned = aligned.reshape(-1, 40) - x_mean

pca = PCA(n_components=3)
pca.fit(x_aligned)

# %%

# 1 std is sqrt of variance
b = np.sqrt(pca.explained_variance_.copy())


def plot_mouth(ax, x, title=None):
    outer = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]
    inner = [12, 13, 14, 15, 16, 17, 18, 19, 12]
    ax.set_title(title)
    ax.scatter(x[:, 0], x[:, 1])
    ax.plot(x[inner, 0], x[inner, 1], '-r', alpha=0.5)
    ax.plot(x[outer, 0], x[outer, 1], '-r', alpha=0.5)
    ax.set_xlim(-60, 60)
    ax.set_ylim(60, -60)
    ax.set_xticks([-50, 0, 50])
    ax.set_yticks([50, 0, -50])
    ax.set_aspect('equal', 'box')


fname = Path("~/tmp/pca").expanduser().resolve()
r = list(np.arange(-3, 3, 0.2)) + list(np.arange(2.8, -3, -0.2))
for j, k in enumerate(r):
    print(j, k)
    fig, axs = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
    fig.suptitle('First 3 Components +/- 3 standard deviations')
    for i in range(3):
        _m = np.zeros(3)
        _m[i] = k
        p = b * _m
        x = (x_mean + pca.inverse_transform(p)).reshape(20, 2)
        title = f'$\sigma$ * ({_m[0]: 0.2f}, {_m[1]: 0.2f}, {_m[2]: 0.2f})'
        plot_mouth(axs[i], x, title)
        if i == 0:
            axs[i].set_ylabel('pixels')
        if i == 1:
            axs[i].set_xlabel('pixels')
    plt.close(fig)
    fig.savefig(fname / f"pca_mouth_{j:04d}.png")

# %%

"""
Make a video:
ffmpeg \
    -framerate 5 \
    -r 5 \
    -pattern_type glob \
    -i '*.png' \
    -vf scale=600:-1 \
    out.gif
"""
