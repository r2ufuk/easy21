from random import random

import numpy as np

from players import base


class Player(base.Player):
    def __init__(self, table_setter, lambada):
        super().__init__(table_setter)

        self.lambada = lambada

        self.coarse_player = [range(i, j + 1) for i, j in ([1, 6], [4, 9], [7, 12], [10, 15], [13, 18], [16, 21])]
        self.coarse_dealer = [range(i, j + 1) for i, j in ([1, 4], [4, 7], [7, 10])]
        self.coarse_action = [i for i in range(self.num_actions)]
        self.feature_shape = self._init_features().shape
        self.weights = np.zeros(self.feature_shape)

    def __call__(self):
        table = self.table_setter()

        state = self._get_state(table)
        action = self._act(state)

        trace = np.zeros(self.feature_shape)

        while True:
            reward = table.step(action)

            state_next = self._get_state(table)

            trace, reward, terminal = self._evaluate(state, action, reward, state_next, trace)

            action_next = self._improve(state_next, reward, trace, terminal)

            if terminal:
                break

            trace *= self.lambada

            state = state_next
            action = action_next

    def get_q(self):
        q = self.generate_q_table()
        return q


    def _evaluate(self, state, action, reward, state_next, trace):
        active_features = self._activate_features(state, action)

        trace[active_features] = 1

        terminal = self._determinate(reward)

        reward = reward.value if terminal else 0
        reward -= self.weights[active_features].sum()

        return trace, reward, terminal

    def _improve(self, state_next, reward, trace, terminal):

        if terminal:
            action_next = None
            step_size = self._get_step_size()
            self.weights += step_size * reward * trace

        else:
            action_next = self._act(state_next)
            self._update_weights(state_next, action_next, reward, trace)

        return action_next

    def _act(self, state):
        e_greed = self._get_greed()

        if random() < e_greed:
            action = np.random.choice(self.num_actions)
        else:
            action_features = self._calc_action_features(state)
            action_values = self._calc_action_values(action_features)

            action = np.argmax(action_values)

        return action

    def _calc_action_features(self, state):
        feature_indices = [self._activate_features(state, action) for action in
                           range(self.num_actions)]

        action_features = np.zeros((self.num_actions,) + self.feature_shape)

        for i_action, action in enumerate(feature_indices):
            action_features[(i_action,) + action] = 1

        return action_features

    def _calc_action_values(self, features):
        features = [a.flatten() for a in features]
        weights = self.weights.flatten()

        action_values = np.zeros(self.num_actions)

        for i_action, action in enumerate(features):
            action_value = np.dot(action, weights)
            action_values[i_action] = action_value

        return action_values

    def _update_weights(self, state, action, reward, trace):
        active_features = self._activate_features(state, action)
        reward += self.weights[active_features].sum()

        self.weights += self._get_step_size() * reward * trace

    def _init_features(self):
        features = np.zeros((len(self.coarse_player), len(self.coarse_dealer), len(self.coarse_action)))
        return features

    def _activate_features(self, state, action):
        state_p, state_d = state

        i_p = [i for i, c in enumerate(self.coarse_player) if state_p in c]
        i_d = [i for i, c in enumerate(self.coarse_dealer) if state_d in c]
        i_a = self.coarse_action.index(action)

        if [] in [i_p, i_d, i_a]:
            raise ValueError("Undefined state-action pair")

        return i_p, i_d, i_a

    @staticmethod
    def _parse_state(state):
        state_p = state.player_total
        state_d = state.dealer_showing

        return state_p, state_d

    @staticmethod
    def _get_step_size():
        return 0.01

    @staticmethod
    def _get_greed():
        return 0.05
