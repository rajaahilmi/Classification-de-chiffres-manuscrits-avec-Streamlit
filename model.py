from sklearn.neural_network import MLPClassifier

def create_model():
    return MLPClassifier(
        hidden_layer_sizes=(128,),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42
    )
