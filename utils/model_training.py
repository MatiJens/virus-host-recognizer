import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def split_train_test_virus_group(path: str):
    data = pd.read_pickle(path)
    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, test_idx = next(splitter.split(data, groups=data["virus_group"]))
    train_df = data.iloc[train_idx]
    test_df = data.iloc[test_idx]

    X_train = np.stack((train_df["embedding"]).values)
    y_train = train_df["label"]

    X_test = np.stack((test_df["embedding"]).values)
    y_test = test_df["label"]

    return X_train, y_train, X_test, y_test
