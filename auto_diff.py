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
            self.dependencies[0].grad += self.power * self.dependencies[0].value * self.grad
    
    def __add__(self, x):
        return Value(self.value + x.value, '+', [self, x])
    
    def __sub__(self, x):
        return Value(self.value - x.value, '-', [self, x])

    def __mul__(self, x):
        return Value(self.value * x.value, '*', [self, x])

    def __div__(self, x):
        return Value(self.value / x.value, '/', [self, x])

    def dfs(self, topsort):
        self.vis = True
        for i in self.dependencies:
            if not i.vis:
                i.dfs(topsort)
        topsort.append(self)

    def backward(self):
        topsort = []
        self.dfs(topsort)
        topsort = topsort[::-1]

        self.grad = 1
        for i in topsort:
            i.backprop()


"""
# Testing auto differentiation

a = Value(2)
b = Value(3)
c = Value(4)
d = Value(1)
e = a * b
y = e * c + e * d

y.backward()

print(f"a = {a.grad}")
print(f"b = {b.grad}")
print(f"c = {c.grad}")
print(f"d = {d.grad}")
print(f"e = {e.grad}")
print(f"y = {y.grad}")
"""