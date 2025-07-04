import pandas as pd
import numpy as np
import joblib
import os

from lightfm import LightFM

# Пути
MODEL_PATH = "models/lightfm_model.pkl"
TEST_EVENTS_PATH = "data/events_test.csv"
OUTPUT_PATH = "models/recommendations_lightfm_top3.csv"

# Загрузка модели и отображений ID
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("LightFM model not found. Сначала обучи модель.")

model, user_id_map, item_id_map = joblib.load(MODEL_PATH)

# Обратные словари для индексов
user_idx_map = {v: k for k, v in user_id_map.items()}
item_idx_map = {v: k for k, v in item_id_map.items()}

# Загрузка тестовых данных
events_test = pd.read_csv(TEST_EVENTS_PATH)

# Уникальные пары visitorid-itemid
pairs = events_test[["visitorid", "itemid"]].drop_duplicates()

# Фильтрация по тем, кто есть в тренировочных отображениях
pairs = pairs[
    pairs["visitorid"].isin(user_id_map) &
    pairs["itemid"].isin(item_id_map)
].copy()

# Преобразуем в индексы
pairs["user_idx"] = pairs["visitorid"].map(user_id_map)
pairs["item_idx"] = pairs["itemid"].map(item_id_map)

# Предсказание оценок
pairs["score"] = model.predict(pairs["user_idx"].values, pairs["item_idx"].values)

# Оставим только нужные столбцы
pairs = pairs[["visitorid", "itemid", "score"]]

# Оставим top-3 рекомендации на пользователя
top3 = (
    pairs.sort_values(by=["visitorid", "score"], ascending=[True, False])
    .groupby("visitorid")
    .head(3)
    .reset_index(drop=True)
)

# Сохраняем
os.makedirs("models", exist_ok=True)
top3.to_csv(OUTPUT_PATH, index=False)
print(f"Рекомендации сохранены в {OUTPUT_PATH}")
