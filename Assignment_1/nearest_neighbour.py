import numpy as np

class NearestNeighbor:
    """Nearest Neighbor implementation.
    """
    def __init__(self):
        pass

    def train(self, X, Y):
        """Training code

        Args:
            X (np.array): input
            Y (np.array): label
        """
        self.train_X = X
        self.train_Y = Y

    def predict(self, X):
        """Prediction

        Args:
            X (np.array): input

        Returns:
            [np.array]: predictions
        """
        N = X.shape[0]
        Y_pred = np.zeros(N,dtype=self.train_Y.dtype)
        for i in range(N):
            #L1 norm distance is used.
            distances = np.sum(np.abs(self.train_X - X[i,:]), axis=1)
            Y_pred[i] = self.train_Y[np.argmin(distances)]

        return Y_pred