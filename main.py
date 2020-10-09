from tqdm import tqdm
import numpy as np
from players.monte_carlo import Player as MonteCarloPlayer
from players.sarsa import Player as SarsaTabularPlayer
from players.approx import Player as SarsaApproxPlayer
from plot import plot_error
from table import Table
from utils import mean_squared_error


def monte_carlo(num_episodes):
    player = MonteCarloPlayer(Table)

    for _ in tqdm(range(num_episodes)):
        player()

    # player.display(num_games=3)
    player.plot_value(plot_path="q-monte_carlo.png")

    return player.q


def sarsa_tabular(num_episodes, lambdas, true_q):
    for lambada in tqdm(lambdas):
        player = SarsaTabularPlayer(Table, lambada)

        errors = []

        for _ in range(num_episodes):
            player()
            errors.append(mean_squared_error(true_q, player.q))

        plot_error(errors)

    # player.display(num_games=3)
        player.plot_value(plot_path="q-sarsa.png")


def sarsa_approximate(num_episodes, lambdas, true_q):
    player = SarsaApproxPlayer(Table, lambada=lambada)

    for _ in tqdm(range(num_episodes)):
        player()

    player.display(num_games=3)


if __name__ == '__main__':
    num_episodes_true = int(5e4)
    num_episodes_bootstrap = int(5e4)

    tabular_lambdas = np.linspace(0, 1, 11)
    tabular_lambdas = (0, 1)
    approx_lambdas = (0, 1)

    q = monte_carlo(num_episodes_true)
    sarsa_tabular(num_episodes_bootstrap, tabular_lambdas, q)
    # sarsa_approximate(num_episodes_bootstrap, tabular_lambdas, q)
