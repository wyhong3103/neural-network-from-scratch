def squared_error(y_pred, y):
    """
    y_pred = nparray of shape (m) that contains the activation values of output layer in the neural network
    y = ground truth of the training example
    """

    y_pred = y_pred[0]


    return (y - y_pred)**2