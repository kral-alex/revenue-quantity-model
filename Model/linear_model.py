import numpy as np

from Processing.time_series import PriceQuantity


class LinearModel:
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

        self.coefficients = None
        self.y_intercept = None

    def train(self):
        self.y_intercept = np.mean(self.y_train)
        self.coefficients = np.linalg.lstsq(self.x_train, self.y_train - self.y_intercept)  # least squares linear regression

    def get_coefficients(self) -> np.ndarray[np.dtype[float]]:
        return self.coefficients
