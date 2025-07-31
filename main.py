import numpy as np
from linear import LinearRegression

x = np.random.rand(10)
y = 2 * x + 7 + (np.random.rand() * 2)

EPOCHS = 100
LEARNING_RATE = 0.001

linear_regression = LinearRegression(x, y)
linear_regression.train(EPOCHS, LEARNING_RATE)