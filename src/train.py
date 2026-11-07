"""Train an Iris classifier and save the model to disk."""

import os
import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_PATH = os.path.join(MODEL_DIR, "iris_model.pkl")


def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    return df, iris.target_names


def train_model(df, test_size=0.2, random_state=42):
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return clf, accuracy


def save_model(model, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    return path


def main():
    df, target_names = load_data()
    print(f"Dataset loaded: {len(df)} samples, {len(target_names)} classes")
    model, accuracy = train_model(df)
    print(f"Model trained — accuracy: {accuracy:.4f}")
    path = save_model(model)
    print(f"Model saved to {path}")
    return model, accuracy


if __name__ == "__main__":
    main()
