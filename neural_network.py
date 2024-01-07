import numpy as np
from auto_diff import Value
from layers.input_layer import Input_Layer

class NN:
    def __init__(self, layers, loss_fn=None, learning_rate=0.001):
        self.layers = layers
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate

        if not callable(loss_fn):
            raise "Loss function must be provided"
        
        if len(layers) == 0 or not isinstance(layers[0], Input_Layer):
            raise "First layer must to be an input layer"

        layers[0].init()
        for i in range(1, len(layers)):
            if isinstance(layers[i], Input_Layer):
                raise "Hidden layer cannot be an input layer"
            layers[i].init(layers[i-1])
    
    def predict(self, X):
        """
        X = a numpy array of shape (m, n) where m is the number of training exampels and n is the number of features
        """
        
        X = np.array([[Value(j) for j in i] for i in X])

        y_pred = []

        for i in range(len(X)):
            for j in range(len(X[0])):
                self.layers[0].a[j] = X[i][j]

            for j in range(1, len(self.layers)):
                self.layers[j].forward_prop(self.layers[j-1])
            
            y_pred.append(self.layers[-1].get_a())
            
        return y_pred
        

    def fit(self, X, y):
        """
        X = array of shape (m, n) where m is the number of training exampels and n is the number of features that contains the training examples
        y = array of shape (m) that contains the ground truth for each training example
        """

        X = np.array([[Value(j) for j in i] for i in X])

        y = np.array([Value(i) for i in y])

        cost = Value(0)
        for i in range(len(X)):
            self.layers[0].a = X[i]

            for j in range(1, len(self.layers)):
                self.layers[j].forward_prop(self.layers[j-1])
            
            cost += self.loss_fn(self.layers[-1].a,y[i])
        
        cost /= Value(len(X))

        cost.zero()
        cost.backward()

        for i in range(1, len(self.layers)):
            self.layers[i].learn(self.learning_rate)