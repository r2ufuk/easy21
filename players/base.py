import numpy as np


class Player:
    def __init__(self, table_setter):
        self.table_setter = table_setter
        self.state_size, self.num_actions = self._inspect_table()

    def __call__(self):
        raise NotImplementedError

    def _act(self, state):
        raise NotImplementedError

    @staticmethod
    def _parse_state(state):
        raise NotImplementedError

    def display(self, num_games=1):
        import pandas as pd

        for _ in range(num_games):
            table = self.table_setter()

            states, actions, reward = self._run_episode(table)

            states = np.array([s(to_list=True) for s in states])[..., :2]
            actions = np.expand_dims(np.array([table.joystick.decode(a).name for a in actions]), 1)

            game = pd.DataFrame(np.concatenate((states, actions), 1))
            game.columns = ("Player", "Dealer", "Action")

            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(game.to_string(index=False))
            print(reward.name, "\n")

    def _run_episode(self, table):
        actions = []
        states = []
        reward = 0
        terminal = False

        while not terminal:
            state = self._get_state(table)

            action = self._act(state)

            reward = table.step(action)

            actions.append(action)
            states.append(state)

            terminal = self._determinate(reward)

        return states, actions, reward

    def _inspect_table(self):
        table = self.table_setter()

        state_size, num_actions = table.inspect()

        return state_size, num_actions

    def _explore(self):
        return np.random.choice(self.num_actions)

    def _get_state(self, table):
        state = table.state()
        return self._parse_state(state)

    @staticmethod
    def _determinate(reward):
        return reward is not None
