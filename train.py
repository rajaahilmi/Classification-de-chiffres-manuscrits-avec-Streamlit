import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from model import create_model

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------- TRAIN MODEL --------
model = create_model()
model.fit(X_train, y_train)

# -------- EVALUATE --------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# -------- SAVE MODEL --------
import joblib
joblib.dump(model, "model_weights.pkl")
