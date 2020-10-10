import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from utils import roundup


def plot_q(value, path=None):
    plt.clf()
    ax = plt.axes(projection='3d')

    dealer = np.arange(1, 11, 1)
    player = np.arange(1, 22, 1)

    plt.xlabel("Dealer Showing", fontsize=10)
    plt.ylabel("Player Total", fontsize=10)

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


def plot_error_by_episode(errors_by_episode, path=None):
    plt.clf()

    for lambada, err in enumerate(errors_by_episode):
        plt.plot(err, label=f"λ={lambada}")

    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("MSE", fontsize=14, rotation="horizontal", ha="right")

    plt.legend()
    plt.tight_layout()

    if path:
        plt.savefig(path)
    else:
        plt.show()


def plot_error_by_lambda(errors, lambdas, checkpoints, path=None):
    path_org = path
    for i, episode_num in enumerate(checkpoints):
        plt.clf()

        episode_num = roundup(episode_num)

        plt.plot(lambdas, errors[i], label=f"episode={roundup(episode_num)}")

        plt.xlabel("λ", fontsize=16)
        plt.ylabel("MSE", fontsize=14, rotation="horizontal", ha="right")

        plt.xticks(lambdas)

        plt.legend()
        plt.tight_layout()

        if path:
            path = path_org.replace(".", f"({episode_num:.0e}).")
            plt.savefig(path)
        else:
            plt.show()
