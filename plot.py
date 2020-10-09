from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def plot(value, path=None):
    ax = plt.axes(projection='3d')

    dealer = np.arange(1, 11, 1)
    player = np.arange(1, 22, 1)

    plt.xticks(np.arange(1, 11, step=9))
    plt.yticks(np.arange(1, 22, step=20))
    ax.set_zticks([-1, 1])

    ax.set_zlim(-1, 1)
    ax.set_ylim(1, 21)
    ax.set_xlim(1, 10)

    x_scale = 10
    y_scale = 21
    z_scale = 4

    dealer, player = np.meshgrid(dealer, player)

    ax.plot_surface(dealer, player, value, cmap="Greens", antialiased=True, rstride=1, cstride=1, lw=0.25,
                    edgecolors="black")

    scale = np.diag([x_scale, y_scale, z_scale, 1.0])
    scale = scale * (1.0 / np.max(scale))
    scale[3, 3] = 0.65

    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), scale)

    if path:
        plt.savefig(path)
    else:
        plt.show()
