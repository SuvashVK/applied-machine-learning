import numpy as np

def mse(y, y_pred):
    """
    Compute Mean Squared Error
    """

    return np.sum((y - y_pred) ** 2) / len(y)