import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

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
    return make_features_numpy(data.values)

# 1. Загрузка и подготовка (на основе твоего baseline)
train_data = pd.read_csv("train.csv")
target = train_data["target"].values
# Предположим, make_features уже написана
features = make_features(train_data) 

# 2. Масштабирование
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 3. Добавляем "мнение" детектора аномалий
# contamination — ожидаемый процент аномалий (у нас их 7 из 3000 ~ 0.002)
iso_forest = IsolationForest(contamination=0.002, random_state=42)
# Обучаем только на нормальных данных (первые 2993 точки)
iso_forest.fit(features_scaled[:2993])

# Получаем оценку аномальности для всех строк
# decision_function дает "степень нормальности": чем меньше число, тем больше аномалия
anomaly_score = iso_forest.decision_function(features_scaled).reshape(-1, 1)

# Объединяем оригинальные признаки с оценкой аномальности
X_train_final = np.hstack([features_scaled, anomaly_score])

# 4. Обучаем CatBoost на стеке признаков
# Рассчитанный вес для редкого класса: 2993 / 7 ≈ 427[cite: 1]
model = CatBoostClassifier(
    iterations=1000,
    depth=6,
    scale_pos_weight=427,
    verbose=0
)
model.fit(X_train_final, target)

def predict(test_data: np.ndarray) -> int:
    # Важно: используем те же шаги, что и при обучении
    test_features = make_features_numpy(test_data)
    test_scaled = scaler.transform(test_features)
    
    # Добавляем оценку аномальности для последней точки
    test_score = iso_forest.decision_function(test_scaled[-1:]).reshape(1, 1)
    
    # Собираем финальный вектор (признаки + аномальность)
    last_row = np.hstack([test_scaled[-1:], test_score])
    
    return int(model.predict(last_row)[0])