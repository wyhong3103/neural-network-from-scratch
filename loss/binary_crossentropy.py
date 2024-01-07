from auto_diff import Value

def binary_crossentropy(y_pred, y):
    """
    y_pred = nparray of shape (m) that contains the activation values of output layer in the neural network
    y = the ground truth of the training example
    """

    y_pred = y_pred[0]

    return -y * (y_pred.log()) -  (Value(1.0)-y) * ((Value(1.0)-y_pred).log())

