import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier

train_data = pd.read_csv("train.csv")

def make_features_numpy(data: np.ndarray) -> np.ndarray:
    shifts = [1, 5, 20, 100]

    all_features = [data]

    for shift in shifts:
        shifted = np.empty_like(data)
        shifted[:shift, :] = 0
        shifted[shift:, :] = data[:-shift, :]
        all_features.append(shifted)

    return np.hstack(all_features)


def make_features(data: pd.DataFrame) -> np.ndarray:
    data = data.copy()
    if "target" in data.columns:
        data = data.drop(columns=["target"])

    # make_features_numpy делает буквально то же самое, но быстрее на numpy
    # features_columns = data.columns

    # shifts = [1, 5, 20, 100]
    # for shift in shifts:
    #     for feature_column in features_columns:
    #         data[f"{feature_column}_shift_{shift}"] = data[feature_column].shift(shift)

    # data = data.fillna(0)
    return make_features_numpy(data.values)

features = make_features(train_data)
target = train_data["target"].values

model = LogisticRegression()
model.fit(features, target)


def predict(test_data: np.ndarray) -> int:
    test_features = make_features_numpy(test_data)

    # Требуется вернуть предсказание только для последней точки
    last_row = test_features[-1].reshape(1, -1)

    preds = model.predict(last_row)[0]
    return preds
