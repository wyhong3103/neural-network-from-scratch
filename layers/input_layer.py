from auto_diff import Value
import numpy as np
import random

class Input_Layer:
    def __init__(self, units):
        self.units = units

    def init(self):
        self.a = np.array(
            [
                Value(random.random()) for _ in range(self.units)
            ]
        )