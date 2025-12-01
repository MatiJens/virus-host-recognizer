from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GroupShuffleSplit

data_path = "data/processed/protbert.pkl"
data = pd.read_pickle(data_path)

X = np.stack(data["embedding"].values)
y = data["label"]
groups = data["virus_group"]

splitter = GroupShuffleSplit(test_size=0.20, n_splits=1, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]


model_rf = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)


model_rf.fit(X_train, y_train)

y_pred_rf = model_rf.predict(X_test)

cm_rf = confusion_matrix(y_test, y_pred_rf)
report_rf = classification_report(y_test, y_pred_rf)

model_xgb = XGBClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.1,
    subsample=1.0,
    colsample_bytree=0.8,
    gamma=0.1,
    scale_pos_weight=24.984,
    n_jobs=-1,
    random_state=42,
    eval_metric="logloss",
)
model_xgb.fit(X_train, y_train)

y_pred_xgb = model_xgb.predict(X_test)

cm_xgb = confusion_matrix(y_test, y_pred_xgb)
report_xgb = classification_report(y_test, y_pred_xgb)

results_path = "normal_results.txt"

with open(results_path, "w", encoding="utf-8") as f:
    f.write("=== Random Forest Results ===\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm_rf))
    f.write("\n-------------------------------\n")
    f.write(report_rf)
    f.write("\n\n")
    f.write("=== XGBoost Results ===\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm_xgb))
    f.write("\n-------------------------------\n")
    f.write(report_xgb)
