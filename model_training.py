from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.svm import SVC

data_path = "data/processed/protbert.pkl"
data = pd.read_pickle(data_path)

X = np.stack(data["embedding"].values)
y = data["label"]
groups = data["virus_group"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


def param_grid_search(model, param_grid: dict, X_train, y_train) -> None:
    grid_search = RandomizedSearchCV(
        estimator=model, param_distributions=param_grid, cv=3, n_jobs=-1, verbose=3
    )

    grid_search.fit(X_train, y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_}")


print("RANDOM FOREST")
rf_model = RandomForestClassifier()
rf_grid = {
    "max_depth": [3, 5, 7, 10],
    "n_estimators": [100, 200, 300, 400, 500],
    "max_features": [10, 20, 30, 40],
    "min_samples_leaf": [1, 2, 4],
}
param_grid_search(rf_model, rf_grid, X_train, y_train)


print("XGBOOST")
xg_model = XGBClassifier(n_estimators=100, objective="binary:logistic", random_state=42)
xg_grid = {
    "max_depth": [3, 5, 7],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "learning_rate": [0.01, 0.1, 0.3],
}
param_grid_search(xg_model, xg_grid, X_train, y_train)


print("SVM")
svm_model = SVC(kernel="rbf", class_weight="balanced", random_state=42)
svm_grid = {
    "C": [0.1, 10, 1000],
    "gamma": [1, 0.01, 0.0001],
    "kernel": ["rbf"],
}
param_grid_search(svm_model, svm_grid, X_train, y_train)
