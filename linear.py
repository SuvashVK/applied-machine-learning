import numpy as np
from evaluation import mse

class LinearRegression:

    def __init__(self, x, y):
        self.weights = {
            "w1": np.random.rand(),
            "w0": np.random.rand()
        }
        
        self.x = x
        self.y = y

    def train(self, epochs, learning_rate):
        """
        For training linear regression using Gradient Descent
        """

        for t in range(epochs):
            y_pred = self.predict()

            mse_error = mse(self.y, y_pred)

            error = y_pred - self.y

            self.weights["w1"] = self.weights["w1"] - learning_rate * np.dot(error, self.x)
            self.weights["w0"] = self.weights["w0"] - learning_rate * np.sum(error)
        
            print(f"New weight: {self.weights['w1']}, new bias: {self.weights['w0']}, MSE: {mse_error}")   
    
    def predict(self):
        """
        A linear function that processes single variable in a data with bias
    
        :returns y: numpy array
        """
        
        w1 = self.weights["w1"]
        w0 = self.weights["w0"]
        
        y_pred = w1 * self.x + w0
    
        return y_pred