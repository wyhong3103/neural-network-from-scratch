import numpy as np
from auto_diff import Value
import random

class FC_Layer:
    def __init__(self, units, activation):
        self.units = units
        self.activation = activation

    def init(self, prev_layer):
        self.w = np.array([[Value(random.uniform(-1, 1)) for _ in range(prev_layer.units)] for _ in range(self.units)])

        self.b = np.array([Value(0) for _ in range(self.units)])

        self.a = np.array([Value(0) for _ in range(self.units)])
    
    def forward_prop(self, prev_layer):

        self.a = self.w @ prev_layer.a + self.b
        
        if (self.activation == 'relu'):
            for i in range(self.units):
                self.a[i] = self.a[i].relu()
        elif (self.activation == 'sigmoid'):
            for i in range(self.units):
                self.a[i] = self.a[i].sigmoid()
    
    def get_w_grad(self):
        return [[j.grad for j in i] for i in self.w]

    def get_b_grad(self):
        return [i.grad for i in self.b]

    def get_a(self):
        return [i.value for i in self.a]
    
    def learn(self, learning_rate):
        for i in range(len(self.w)):
            for j in range(len(self.w[i])):
                self.w[i][j].value -= learning_rate * self.w[i][j].grad
    
        for i in range(len(self.b)):
            self.b[i].value -= learning_rate * self.b[i].grad