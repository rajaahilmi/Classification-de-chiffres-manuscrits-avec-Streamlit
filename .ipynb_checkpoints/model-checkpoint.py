import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.W1 = np.random.randn(400, 128) * 0.01
        self.b1 = np.zeros((1, 128))
        self.W2 = np.random.randn(128, 10) * 0.01
        self.b2 = np.zeros((1, 10))

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ self.W2 + self.b2
        e = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)
