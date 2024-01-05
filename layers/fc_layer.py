from auto_diff import Value
import numpy as np
import random

class FC_Layer:
    def __init__(self, units, activation):
        self.units = units
        self.activation = activation

    def init(self, prev_layer):
        self.w = np.array(
            [
                [Value(random.random()) for _ in range(prev_layer.units)]
                for _ in range(self.units)
            ]
        )

        self.b = np.array(
            [
                Value(random.random()) for _ in range(self.units)
            ]
        )

        self.a = np.array(
            [
                Value(random.random()) for _ in range(self.units)
            ]
        )
    
    def forward_prop(self, prev_layer):
        self.a = self.w @ prev_layer.a + self.b
        if (self.activation == 'relu'):
            for i in self.a:
                i = i.relu()
        elif (self.activation == 'sigmoid'):
            for i in self.a:
                i = i.sigmoid()
    
    def learn(self, learning_rate):
        for i in self.w:
            for j in i:
                j.value -= learning_rate * j.grad
    
        for i in self.b:
            i.value -= learning_rate * i.grad
