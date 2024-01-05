import math

class Value:
    def __init__(self, value, operator=None, dependencies=[], power = 0):
        self.value = value
        self.operator = operator
        self.dependencies = dependencies
        self.power = power
        self.grad = 0
        self.vis = False

    
    def zero(self):
        self.grad = 0
        for i in self.dependencies:
            i.zero()
    
    def backprop(self):
        if self.operator == '+':
            self.dependencies[0].grad += self.grad
            self.dependencies[1].grad += self.grad
        elif self.operator == '-':
            self.dependencies[0].grad += self.grad
            self.dependencies[1].grad += -self.grad
        elif self.operator == '*':
            self.dependencies[0].grad += self.dependencies[1].value * self.grad
            self.dependencies[1].grad += self.dependencies[0].value * self.grad
        elif self.operator == '/':
            self.dependencies[0].grad += (1/self.dependencies[1].value) * self.grad
            self.dependencies[1].grad += (-self.dependencies[0].value/self.dependencies[1].value**2) * self.grad
        elif self.operator == '^':
            self.dependencies[0].grad += self.power * (self.dependencies[0].value ** (self.power - 1)) * self.grad
        elif self.operator == 'relu':
            self.dependencies[0].grad += self.grad if self.dependencies[0].value > 0 else 0
        elif self.operator == 'sigmoid':
            # Derivative of the sigmoid function
            # https://math.stackexchange.com/a/1225116
            self.dependencies[0].grad += (1 / (1 + math.exp(-self.dependencies[0].value))) * (1 - 1 / (1 + math.exp(-self.dependencies[0].value))) * self.grad
    
    def __add__(self, x):
        return Value(self.value + x.value, '+', [self, x])
    
    def __sub__(self, x):
        return Value(self.value - x.value, '-', [self, x])

    def __mul__(self, x):
        return Value(self.value * x.value, '*', [self, x])

    def __truediv__(self, x):
        return Value(self.value / x.value, '/', [self, x])
    
    def relu(self):
        return Value(max(0, self.value), 'relu', [self])

    def sigmoid(self):
        return Value(1 / (1 + math.exp(-self.value)), 'sigmoid', [self])

    def dfs(self, topsort):
        self.vis = True
        for i in self.dependencies:
            if not i.vis:
                i.dfs(topsort)
        topsort.append(self)

    def backward(self):
        topsort = []
        # Run a DFS from the result node, generate the topological order
        # Run backprop from the topological order
        self.dfs(topsort)

        # A reverse topological order of a graph is equivalent to the topological order of the graph with reverse edges
        # Proof: https://qr.ae/pKV2Zn
        topsort = topsort[::-1]

        self.grad = 1
        for i in topsort:
            i.backprop()


"""
# Simple equation 1

a = Value(2)
b = Value(3)
c = Value(4)
d = Value(1)
e = a * b
y = e * c + e * d

y.backward()
"""

"""
# Simple equation 2

a = Value(2)
b = Value(3)
y = a / b

y.backward()
"""

"""
# Simple neural network

x = Value(2)
w = [
    [
        [Value(2)]
    ],
    [
        [Value(3)],
        [Value(4)]
    ],
    [
        [Value(10), Value(11)]
    ]
]
b = [
    [Value(10)],
    [Value(-5), Value(-3)],
    [Value(3)]
]

a = [
    [None],
    [None, None],
    [None]
]

a[0][0] = w[0][0][0] * x + b[0][0]
a[1][0] = w[1][0][0] * a[0][0] + b[1][0]
a[1][1] = w[1][1][0] * a[0][0] + b[1][1]
a[2][0] = w[2][0][0] * a[1][0] + w[2][0][1] * a[1][1] + b[2][0]

a[2][0].backward()
"""