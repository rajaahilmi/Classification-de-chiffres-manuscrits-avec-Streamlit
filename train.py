import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from model import NeuralNetwork

# -------- LOAD DATA --------
data1 = loadmat("data/ex3data1.mat")
data2 = loadmat("data/ex4data1.mat")

X1, y1 = data1["X"], data1["y"]
X2, y2 = data2["X"], data2["y"]

y1 = y1.flatten()
y2 = y2.flatten()
y1[y1 == 10] = 0
y2[y2 == 10] = 0

X = np.vstack((X1, X2)) / 255.0
y = np.hstack((y1, y2))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# -------- MODEL --------
model = NeuralNetwork()
lr = 0.1
epochs = 50

# -------- TRAINING --------
for epoch in range(epochs):

    # Forward
    z1 = X_train @ model.W1 + model.b1
    a1 = np.maximum(0, z1)
    z2 = a1 @ model.W2 + model.b2

    exp = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
    probs = exp / np.sum(exp, axis=1, keepdims=True)

    # Backprop
    probs[np.arange(len(y_train)), y_train] -= 1
    probs /= len(y_train)

    dW2 = a1.T @ probs
    db2 = np.sum(probs, axis=0, keepdims=True)

    da1 = probs @ model.W2.T
    dz1 = da1 * (z1 > 0)

    dW1 = X_train.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    # Update
    model.W2 -= lr * dW2
    model.b2 -= lr * db2
    model.W1 -= lr * dW1
    model.b1 -= lr * db1

    if epoch % 10 == 0:
        print(f"Epoch {epoch}")

# -------- TEST --------
z1 = X_test @ model.W1 + model.b1
a1 = np.maximum(0, z1)
z2 = a1 @ model.W2 + model.b2

preds = np.argmax(z2, axis=1)
accuracy = np.mean(preds == y_test)

print("Accuracy:", accuracy)

# -------- SAVE --------
np.savez(
    "model_weights.npz",
    W1=model.W1,
    b1=model.b1,
    W2=model.W2,
    b2=model.b2
)
