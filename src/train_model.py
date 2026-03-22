import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def load_data():
    data = pd.read_csv("data/dataset.csv", header=None)
    
    X = data.iloc[:, :-1].values  # features
    y = data.iloc[:, -1].values   # labels
    
    return X, y


def train():
    X, y = load_data()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = RandomForestClassifier(n_estimators=100)

    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {acc * 100:.2f}%")

    # Save model
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model saved successfully!")


if __name__ == "__main__":
    train()