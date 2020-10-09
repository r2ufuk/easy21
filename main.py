from tqdm import tqdm

from players.monte_carlo import Player as MonteCarloPlayer
from table import Table


def main(player, num_episodes):
    for _ in tqdm(range(num_episodes)):
        player()

    player.display(num_games=3)
    player.plot_value()


if __name__ == '__main__':
    # p = SarsaPlayer(Table, 0.1)
    p = MonteCarloPlayer(Table)
    n = 10000

    main(p, n)
