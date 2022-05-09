import numpy as np
import matplotlib.pyplot as plt


class BatchGradientDescent:
    def __init__(self, learning_rate=0.0001, epochs=100000):
        # Intialize Learning rate
        self.learning_rate = learning_rate

        # Intialize Number of iterations
        self.epochs = epochs

    def fit(self, X, y):

        # Get number of features
        n = X.shape[1]

        # Get number of samples
        m = X.shape[0]

        # Add X0 = 1 to each instance
        X_b = np.c_[np.ones((m, 1)), X]

        # Intialize random model parameters
        self.weights = np.ones((X_b.shape[1], 1))

        for epoch in range(self.epochs):
            y_pred = X_b.dot(self.weights)
            gradients = 2/m * X_b.T.dot(y_pred - y)
            self.weights = self.weights - self.learning_rate * gradients

        print("Weights", self.weights)


if __name__ == "__main__":
    # Generating Fake Data
    X = 2 * np.random.rand(100, 1)
    y = 1 + 3 * X + np.random.randn(100, 1)

    model = BatchGradientDescent()
    model.fit(X, y)
