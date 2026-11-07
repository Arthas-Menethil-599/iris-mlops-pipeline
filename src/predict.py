"""Load a trained model and make predictions."""

import os
import joblib
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "iris_model.pkl")
TARGET_NAMES = ["setosa", "versicolor", "virginica"]


def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}. Run train.py first.")
    return joblib.load(path)


def predict(model, features):
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    probabilities = model.predict_proba(features)
    return {
        "class_id": int(prediction[0]),
        "class_name": TARGET_NAMES[prediction[0]],
        "probabilities": {
            name: float(prob)
            for name, prob in zip(TARGET_NAMES, probabilities[0])
        },
    }


def main():
    model = load_model()
    sample = [5.1, 3.5, 1.4, 0.2]
    result = predict(model, sample)
    print(f"Input: {sample}")
    print(f"Predicted class: {result['class_name']} (id={result['class_id']})")
    print(f"Probabilities: {result['probabilities']}")


if __name__ == "__main__":
    main()
