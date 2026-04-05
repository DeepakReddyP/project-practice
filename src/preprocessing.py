# src/preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def preprocess():
    # Load raw data
    df = pd.read_csv("data/raw/heart.csv")

    # -------- ENCODING --------
    dataset = pd.get_dummies(
        df,
        columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    )

    # -------- SCALING --------
    scaler = StandardScaler()
    columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

    dataset[columns_to_scale] = scaler.fit_transform(dataset[columns_to_scale])

    # -------- SPLIT --------
    X = dataset.drop('target', axis=1)
    y = dataset['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------- SAVE --------
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    print("✅ Preprocessing completed and files saved")


if __name__ == "__main__":
    preprocess()