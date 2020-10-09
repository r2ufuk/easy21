from random import random

import numpy as np

from players import base
from plot import plot_q
from utils import Return


class Player(base.Player):
    def __init__(self, table_setter, n_0=100):
        super().__init__(table_setter)
        self.n_0 = n_0

        self.pi = self._generate_arbitrary_policy()
        self.q = self._generate_arbitrary_state_action_pairs()
        self.g = self._init_returns()
        self.visits = np.zeros(self.q.shape, dtype=np.int)

    def __call__(self):
        table = self.table_setter()

        states, actions, reward = self._run_episode(table)

        for idx, (state, action) in enumerate(zip(states, actions)):
            if state in states[:idx] and action in actions[:idx]:
                continue

            self._evaluate(state, action, reward)
            self._improve(state)

    def _act(self, state):
        e_greed = self._get_greed(state)

        if random() < e_greed:
            action = self._explore()
        else:
            action = self._exploit(state)

        idx = self._index(state, action)
        self.visits[idx] += 1

        return action

    @staticmethod
    def _parse_state(state):
        state_p = state.player_total - 1
        state_d = state.dealer_showing - 1

        return state_p, state_d

    def plot_value(self, plot_path=None):
        optimal_value = np.max(self.q, 2)

        plot_q(optimal_value, plot_path)

    def _evaluate(self, state, action, reward):
        idx = self._index(state, action)

        self.g[idx].update(reward)

        step_size = self._get_step_size(state, action)
        avg_return = self.g[idx]()

        self.q[idx] += step_size * (avg_return - self.q[idx])

    def _improve(self, state):
        best_action = np.argmax(self.q[state])
        self.pi[state] = best_action

    def _exploit(self, state):
        return self.pi[state]

    def _generate_arbitrary_policy(self):
        policy = np.zeros(self.state_size, dtype=np.int)
        for i, _ in np.ndenumerate(policy):
            policy[i] = np.random.choice(self.num_actions)

        return policy

    def _generate_arbitrary_state_action_pairs(self):
        return np.zeros(self.state_size + (self.num_actions,))

    def _init_returns(self):
        returns = np.empty(self.q.shape, dtype=object)
        for i, _ in np.ndenumerate(returns):
            returns[i] = Return()

        return returns

    def _get_step_size(self, state, action):
        idx = self._index(state, action)
        return 1 / self.visits[idx]

    def _get_greed(self, state):
        n_s = np.sum(self.visits[state]) + 1  # Account for current state

        return self.n_0 / (self.n_0 + n_s)

    @staticmethod
    def _index(state, action):
        return state + (action,)
