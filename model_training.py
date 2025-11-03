from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

data_path = "data/processed/concated_data.pkl"

data = pd.read_pickle(data_path)

X = np.stack(data["embedding"].values)
y = data["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)


model_rf = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)

model_rf.fit(X_train, y_train)

y_pred = model_rf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

results_path = "results.txt"

with open(results_path, "w", encoding="utf-8") as f:
    f.write("Confusion Matrix:\n")
    f.write(str(cm))
    f.write("\n===============================\n")
    f.write(report)
