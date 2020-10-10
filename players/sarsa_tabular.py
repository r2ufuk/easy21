import numpy as np

from players import monte_carlo


class Player(monte_carlo.Player):
    def __init__(self, table_setter, lambada):
        super().__init__(table_setter)

        self.lambada = lambada

    def __call__(self):
        table = self.table_setter()

        state = self._get_state(table)
        action = self._act(state)

        trace = np.zeros(self.q.shape)

        while True:
            reward = table.step(action)

            state_next = self._get_state(table)

            action, trace, terminal = self._evaluate(state, action, reward, state_next, trace)

            self._improve(state)

            state = state_next

            if terminal:
                break

    def _evaluate(self, state, action, reward, state_next, trace):
        idx = self._index(state, action)

        trace[idx] = 1  # Replacing

        value = self.q[idx]

        terminal = self._determinate(reward)

        if terminal:
            reward = reward.value
            action_next = None
            value_bootstrap = 0

        else:
            reward = 0
            action_next = self._act(state_next)
            idx_next = self._index(state_next, action_next)
            value_bootstrap = self.q[idx_next]

        step_size = self._get_step_size(state, action)

        adjustment = reward + value_bootstrap - value

        self.q += adjustment * step_size * trace

        trace *= self.lambada

        return action_next, trace, terminal
