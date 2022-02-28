# %%
import matplotlib.pyplot as plt


def set_style():
    """set the style of the plots"""
    plt.style.use('fivethirtyeight')
    plt.rcParams['figure.figsize'] = [9.0, 5.0]
    plt.rcParams['figure.dpi'] = 240
    plt.rcParams['axes.linewidth'] = 1.25
    plt.rcParams['savefig.pad_inches'] = 0.2
    plt.rcParams['savefig.bbox'] = 'tight'


set_style()

recall = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
p1 = [1, 1, 1, 1, 1, 1, 1, 1, 0.7, 0.5, 0.3]
p2 = [1, 1, 1, 1, 0.8, 0.85, 0.81, 0.65, 0.5, 0.2, 0.1]

fig, ax = plt.subplots(1, 1)
ax.set_xlabel("Recall (True Positive Rate)")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve")
ax.plot(recall, p1, '-X', label="Model 1")
ax.plot(recall, p2, '-X', label="Model 2")
ax.legend()
fig.savefig("pr_curves.png")

# %%
