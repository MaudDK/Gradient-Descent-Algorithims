import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class BatchGradientDescent:
    def __init__(self, learning_rate=0.01, epochs=10000, random_state=None):
        # Intialize Learning rate
        self.learning_rate = learning_rate

        # Intialize Number of iterations
        self.epochs = epochs

        # Sets Model Random Seed
        np.random.seed(random_state)

    def fit(self, X, y, verbose=False):

        # Get number of features
        n = X.shape[1]

        # Get number of samples
        m = X.shape[0]

        # Add X0 = 1 to each instance
        X_b = np.c_[np.ones((m, 1)), X]

        # Intialize random model parameters
        self.weights = np.random.rand(X_b.shape[1], 1)

        for epoch in range(1, self.epochs+1):

            # Predict y using weights
            y_pred = X_b.dot(self.weights)

            # Calculate gradients
            gradients = 2/m * X_b.T.dot(y_pred - y)

            # Modify weights in steepest descent
            self.weights = self.weights - self.learning_rate * gradients

            if verbose:
                # Display Epoch Cost
                if epoch % 1000 == 0:
                    # Calculate RMSE
                    cost = self.cost(y, y_pred)
                    print(f'Train Epoch: {epoch} | RMSE: {cost}')

    def predict(self, X):
        # Get number of samples
        m = X.shape[0]

        # Add X0 = 1 to each instance
        X_b = np.c_[np.ones((m, 1)), X]

        # Predict y using weights
        y_pred = X_b.dot(self.weights)

        return y_pred

    def cost(self, y, y_pred):
        # Calculate RMSE
        cost = np.sqrt(np.mean(np.square(y_pred-y)))
        return cost


if __name__ == "__main__":
    # Generating Fake Data with some noise
    X = 2 * np.random.rand(1000, 1)
    y = 2 + 3 * X + np.random.randn(1000, 1)

    # Splitting data to train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Initializing Model
    model = BatchGradientDescent()

    # Training Model
    model.fit(X_train, y_train, verbose=True)

    # Predicting on Test
    y_pred = model.predict(X_test)

    # Calculating RMSE
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    # Comparing Model RMSE and Sklearn RMSE (SHOULD BE SAME THING)
    print(
        f'Sklearn RMSE: {rmse}\nModel   RMSE: {model.cost(y_test, y_pred)}')

    # Displaying Model Weights
    print(model.weights)
