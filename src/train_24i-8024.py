"""
MLOps Assignment 1 - House Price Prediction
Student ID: 24i-8024

Loads the dataset from data/, trains a model and saves it into model/.
Run from the project root:  python src/train_24i-8024.py
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

STUDENT_ID = "24i-8024"

# Paths are resolved from the project root so the script works from any folder
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, "data", "dataset.csv")
MODEL_DIR = os.path.join(ROOT_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, f"model_{STUDENT_ID}.pkl")

# Hyperparameters
RANDOM_STATE = 42
N_ESTIMATORS = 200


def create_sample_dataset(path):
    """data/ is not tracked by Git, so build a sample house price dataset
    when no dataset.csv is present (for example on a fresh clone)."""
    rng = np.random.default_rng(RANDOM_STATE)
    n = 1000
    df = pd.DataFrame({
        "area_sqft": rng.integers(500, 4500, n),
        "bedrooms": rng.integers(1, 6, n),
        "bathrooms": rng.integers(1, 4, n),
        "age_years": rng.integers(0, 40, n),
        "distance_to_city_km": rng.uniform(1, 30, n).round(1),
    })
    price = (
        50_000
        + 120 * df["area_sqft"]
        + 15_000 * df["bedrooms"]
        + 10_000 * df["bathrooms"]
        - 1_200 * df["age_years"]
        - 2_500 * df["distance_to_city_km"]
        + rng.normal(0, 20_000, n)
    )
    df["price"] = price.round(0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[{STUDENT_ID}] No dataset found, created a sample one at {path}")


def load_data():
    if not os.path.exists(DATA_PATH):
        create_sample_dataset(DATA_PATH)
    print(f"[{STUDENT_ID}] Loading dataset from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"[{STUDENT_ID}] Loaded {df.shape[0]} rows and {df.shape[1]} columns")
    return df


def train(df):
    X = df.drop(columns=["price"])
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    scaler = preprocessing.MinMaxScaler()  # scale features to the 0-1 range

    model = GradientBoostingRegressor(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE
    )

    steps = [("model", model)]
    if scaler is not None:
        steps.insert(0, ("scaler", scaler))
    pipeline = Pipeline(steps)

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    print(f"[{STUDENT_ID}] MAE: {mean_absolute_error(y_test, predictions):,.0f}")
    print(f"[{STUDENT_ID}] R2 : {r2_score(y_test, predictions):.3f}")
    return pipeline


def save_model(pipeline):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"[{STUDENT_ID}] Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    data = load_data()
    trained = train(data)
    save_model(trained)
