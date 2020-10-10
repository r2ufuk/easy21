import math
import os

import numpy as np
from tqdm import tqdm

from players.monte_carlo import Player as MonteCarloPlayer
from players.sarsa_tabular import Player as SarsaTabularPlayer
from players.sarsa_approx import Player as SarsaApproxPlayer
from plot import plot_error_by_lambda, plot_error_by_episode
from table import Table
from utils import mean_squared_error


def monte_carlo(num_episodes, plot_root, q_npy_path=None, ):
    player = MonteCarloPlayer(Table)

    if q_npy_path and os.path.isfile(q_npy_path):
        print("Loading from file")
        player.q = np.load(q_npy_path)

    else:
        for _ in tqdm(range(num_episodes)):
            player()

        if q_npy_path:
            np.save("q.npy", player.q)

    os.makedirs(plot_root, exist_ok=True)

    player.plot_value(path=os.path.join(plot_root, f"Q[{num_episodes:.0e}].png"))

    return player.q


def sarsa_driver(player_constructor, num_episodes, lambdas, true_q, plot_root):
    if isinstance(num_episodes, list):
        checkpoints = num_episodes
        num_episodes = num_episodes[-1]
    else:
        checkpoints = [int(10 ** (i + 3)) for i in range(int(math.log10(num_episodes / 1e3)) + 1)]

    lambda_errors = np.empty((len(checkpoints), 0)).tolist()

    learning_curves = []

    for lambada in tqdm(lambdas):
        player = player_constructor(Table, lambada)

        episode_errors = []

        for i_episode in range(num_episodes):
            player()
            episode_errors.append(mean_squared_error(true_q, player.get_q()))

            episode_num = i_episode + 1
            if episode_num in checkpoints:
                lambda_errors[checkpoints.index(episode_num)].append(np.mean(episode_errors))

        if lambada in (0, 1):
            learning_curves.append(episode_errors)

    os.makedirs(plot_root, exist_ok=True)

    path = os.path.join(plot_root, f"learning_curves.png")
    plot_error_by_episode(learning_curves, path=path)

    path = os.path.join(plot_root, f"mse.png")
    plot_error_by_lambda(lambda_errors, lambdas, checkpoints, path=path)


if __name__ == '__main__':
    num_episodes_true = int(1e6)
    num_episodes_bootstrap = [int(i) for i in [1e3, 2e4]]

    tabular_lambdas = np.linspace(0, 1, 11)
    approx_lambdas = (0, 1)

    monte_carlo_q_path = "q.npy"

    plot_root_monte_carlo = "plots/monte_carlo"
    plot_root_sarsa_tabular = "plots/sarsa/tabular"
    plot_root_sarsa_approx = "plots/sarsa/approx"

    print("-*- Monte Carlo -*-")
    q = monte_carlo(num_episodes_true, plot_root_monte_carlo, monte_carlo_q_path)

    # print("-*- Sarsa Tabular-*-")
    # sarsa_driver(SarsaTabularPlayer, num_episodes_bootstrap, tabular_lambdas, q, plot_root_sarsa_tabular)

    print("-*- Sarsa Approx-*-")
    sarsa_driver(SarsaApproxPlayer, num_episodes_bootstrap, tabular_lambdas, q, plot_root_sarsa_approx)
