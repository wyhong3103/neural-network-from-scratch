from auto_diff import Value

class Input_Layer:
    def __init__(self, units):
        self.units = units

    def init(self):
        self.a = [Value(0) for _ in range(self.units)]