import numpy as np

class NearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, Y):
        self.train_X = X
        self.train_Y = Y

    def predict(self, X):
        N = X.shape[0]
        Y_pred = np.zeros(N,dtype=self.train_Y.dtype)
        for i in range(N):
            distances = np.sum(np.abs(self.train_X - X[i,:]), axis=1)
            Y_pred[i] = self.train_Y[np.argmin(distances)]

        return Y_pred