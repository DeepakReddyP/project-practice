# src/train_model.py

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def load_data():
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    return X_train, y_train


def train_knn(X, y):
    print("\n🔹 KNN Results")
    for i in range(1, 21):
        model = KNeighborsClassifier(n_neighbors=i)
        score = cross_val_score(model, X, y, cv=10).mean()
        print(f"K={i}, Accuracy={round(score,3)}")

    model = KNeighborsClassifier(n_neighbors=12)
    score = cross_val_score(model, X, y, cv=10).mean()
    print(f"✅ Final KNN Accuracy (K=12): {round(score*100,2)}%")


def train_decision_tree(X, y):
    print("\n🔹 Decision Tree Results")
    for i in range(1, 11):
        model = DecisionTreeClassifier(max_depth=i)
        score = cross_val_score(model, X, y, cv=10).mean()
        print(f"Depth={i}, Accuracy={round(score,3)}")

    model = DecisionTreeClassifier(max_depth=3)
    score = cross_val_score(model, X, y, cv=10).mean()
    print(f"✅ Final Decision Tree Accuracy: {round(score*100,2)}%")


def train_random_forest(X, y):
    print("\n🔹 Random Forest Results")
    for i in range(10, 101, 10):
        model = RandomForestClassifier(n_estimators=i)
        score = cross_val_score(model, X, y, cv=5).mean()
        print(f"Trees={i}, Accuracy={round(score,3)}")

    model = RandomForestClassifier(n_estimators=90)
    score = cross_val_score(model, X, y, cv=5).mean()
    print(f"✅ Final Random Forest Accuracy: {round(score*100,2)}%")


def main():
    X, y = load_data()

    train_knn(X, y)
    train_decision_tree(X, y)
    train_random_forest(X, y)


if __name__ == "__main__":
    main()