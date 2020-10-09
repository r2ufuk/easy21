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

        state = table.state()
        state_p, state_d = self._parse_state(state)
        action = self._act(state_p, state_d)

        egilibilibity_trace = np.zeros(self.feature_shape)

        while True:
            active_features = self._activate_features(state_p, state_d, action)

            egilibilibity_trace[active_features] = 1

            reward = table.step(action)

            state_next = table.state()
            state_p_next, state_d_next = self._parse_state(state_next)

            terminal = self._determinate(reward)

            reward = reward.value if terminal else 0
            reward -= self.weights[active_features].sum()

            if terminal:
                step_size = self._get_step_size()
                self.weights += step_size * reward * egilibilibity_trace
                break

            action_next = self._act(state_p_next, state_d_next)

            self._update_weights(state_p_next, state_d_next, action_next, reward, egilibilibity_trace)

            egilibilibity_trace *= self.lambada

            state_p, state_d = state_p_next, state_d_next
            action = action_next

    def _update_weights(self, state_p, state_d, action, reward, egilibilibity_trace):
        active_features = self._activate_features(state_p, state_d, action)
        reward += self.weights[active_features].sum()

        self.weights += self._get_step_size() * reward * egilibilibity_trace

    def _act(self, state_p, state_d):
        e_greed = self._get_greed()

        if random() < e_greed:
            action = np.random.choice(self.num_actions)
        else:
            feature_indices = [self._activate_features(state_p, state_d, action) for action in
                               range(self.num_actions)]

            action_features = np.zeros((self.num_actions,) + self.feature_shape)
            for i_action, action in enumerate(feature_indices):
                action_features[(i_action,) + action] = 1

            action_features_flat = [a.flatten() for a in action_features]

            action_values = np.zeros(self.num_actions)
            for i_action, action_flat in enumerate(action_features_flat):
                action_value = np.dot(action_flat, self.weights.flatten())
                action_values[i_action] = action_value

            action = np.argmax(action_values)

        return action

    @staticmethod
    def _parse_state(state):
        state_p = state.player_total
        state_d = state.dealer_showing

        return state_p, state_d

    def _init_features(self):
        features = np.zeros((len(self.coarse_player), len(self.coarse_dealer), len(self.coarse_action)))
        return features

    def _activate_features(self, state_p, state_d, action):
        i_p = [i for i, c in enumerate(self.coarse_player) if state_p in c]
        i_d = [i for i, c in enumerate(self.coarse_dealer) if state_d in c]
        i_a = self.coarse_action.index(action)

        return i_p, i_d, i_a

    @staticmethod
    def _get_step_size():
        return 0.01

    @staticmethod
    def _get_greed():
        return 0.05
