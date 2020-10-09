from enum import Enum
import random


class Color(Enum):
    black = 1
    red = -1


class Deck:
    def __init__(self, value_min=1, value_max=10, black_prob=2/3):
        self.value_min = value_min
        self.value_max = value_max
        self.black_prob = black_prob

    def draw(self, first=False):
        value = random.randint(1, 10)

        if not first:
            color = Color.black if random.random() <= self.black_prob else Color.red
            value *= color.value

        return value
