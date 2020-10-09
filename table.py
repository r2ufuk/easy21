from copy import copy
import pandas as pd
from enum import Enum
from deck import Deck


class Reward(Enum):
    lost: int = -1
    draw: int = 0
    won: int = 1


class Action(Enum):
    stick: int = 0
    hit: int = 1

    @classmethod
    def decode(cls, value):
        enumap = cls._value2member_map_

        if isinstance(value, cls):
            return value
        elif value in enumap:
            return enumap[value]
        else:
            raise AttributeError("Invalid action")


class State:
    def __init__(self, player_total, dealer_showing):
        self.player_total = player_total
        self.dealer_total = dealer_showing
        self.dealer_showing = dealer_showing

    def __call__(self, to_list=False):
        if to_list:
            x = [self.player_total, self.dealer_total, self.dealer_showing]
        else:
            x = self

        return copy(x)


class Table:
    def __init__(self, valid_min=1, valid_max=21):
        self.valid_min = valid_min
        self.valid_max = valid_max

        self.joystick = Action
        self.deck = Deck()
        self.state = State(self.deck.draw(first=True), self.deck.draw(first=True))

    def step(self, action):
        action = Action.decode(action)
        if action is Action.hit:
            self.state.player_total += self.deck.draw()

            if self._bust(self.state.player_total):
                return Reward.lost
            else:
                return None

        elif action is Action.stick:
            return self._deal()

    def inspect(self):
        num_states_p = self.valid_max - self.valid_min + 1
        num_states_d = self.deck.value_max - self.deck.value_min + 1

        num_actions = len(self.joystick)

        state_size = (num_states_p, num_states_d)

        return state_size, num_actions

    def _deal(self, sticky_spot=17):
        while self.state.dealer_total < sticky_spot:
            self.state.dealer_total += self.deck.draw()

            if self._bust(self.state.dealer_total):
                return Reward.won

        if self.state.dealer_total < self.state.player_total:
            return Reward.won
        if self.state.dealer_total > self.state.player_total:
            return Reward.lost
        else:
            return Reward.draw

    def _bust(self, value):
        if value not in range(self.valid_min, self.valid_max + 1):
            return True
        else:
            return False
