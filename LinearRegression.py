from matplotlib import pyplot as plt
import numpy as np


class LinearRegression():
    def __init__(self, learning_rate=0.1, iters=10000, show_cost=False, show_per_iter=1000):
        """
        Linear Regression Class Constructor
        Initializes learning rate, training iterations
        Show cost parameter determines whether to print cost while training
        Show per iteration parameter determines the interval of displaying cost per each iteration
        """
        self.learning_rate = learning_rate
        self.iters = iters
        self.show_cost = show_cost
        self.show_per_iter = show_per_iter

    def predict(self, theta, X):
        """
        yhat = theta(0)*x(0) + ... + theta(n)*x(n)

            or
        yhat = theta ⋅ (dot) X

        theta(θ) is the model's parameter vector, containing the bias term θ0 and the feature weights θ1 to θn
        x is the instance's feature vector, containing x0 to xn, with x0 always equal to 1.
        yhat is the hypothesis function using the parameters theta
        """
        yhat = np.dot(X, theta)
        return yhat

    def cost(self, y, yhat):
        """
        Mean Squared Error Function For Cost
        Determines the cost for the function
        """
        mse = 1/self.m * np.sum(np.power(yhat-y, 2))
        return mse

    def train(self, X, y):
        """
        Trains weighs theta to minimize cost based on MSE using gradient descent
        """

        # Transforms dataset to add x0 = 1 to each instance
        X = np.c_[np.ones((X.shape[0], 1)), X]

        # Initializes weights vector to zeros
        self.theta = np.zeros((X.shape[1], 1))
        self.m = X.shape[0]

        # Begins training using gradient descent
        for i in range(self.iters + 1):
            yhat = self.predict(self.theta, X)
            cost = self.cost(y, yhat)

            if i % self.show_per_iter == 0 and self.show_cost is True:
                print(f'Cost at {i}: {cost}')

            # Gradient Descent
            gradients = 2/self.m * X.T.dot(yhat - y)

            # Updates Weights based on learning rate and gradient
            self.theta = self.theta - self.learning_rate * gradients

        return self.theta


if __name__ == '__main__':

    # Generate Random Data
    X = 2 * np.random.rand(100, 3)
    y = 4 + 3 * X + np.random.randn(100, 3)

    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.axis([0, 2, 0, 15])

    # Inv Method
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # add x0 = 1 to each instance
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    # Gradient Descent Linear Regression
    model = LinearRegression(learning_rate=0.01, iters=10000, show_cost=True)
    theta = model.train(X, y)

    print(
        f'My Theta: {theta}\nBest Theta: {theta_best}')

    # NOTE SAME THING
    ypred = X*theta[1] + theta[0]
    ypred = np.dot(X_b, theta)

    bestpred = np.dot(X_b, theta_best)

    plt.plot(X, y, "b.")
    plt.plot(X, bestpred, "g-")
    plt.plot(X, ypred, "r-")
    plt.show()
