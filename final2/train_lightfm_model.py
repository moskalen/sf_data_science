import pandas as pd
import numpy as np
import scipy.sparse as sp
from lightfm import LightFM
import joblib
import os

# Пути
EVENTS_PATH = "data/events_train.csv"
MODEL_PATH = "models/lightfm_model_best.pkl"

# Пропуск, если модель уже существует
if os.path.exists(MODEL_PATH):
    print("LightFM model already exists. Skipping training.")
    exit()

# Загрузка данных
events = pd.read_csv(EVENTS_PATH)

# Фильтруем только покупки
purchases = events[events["event"] == "transaction"]

# Уникальные ID
user_ids = purchases["visitorid"].unique()
item_ids = purchases["itemid"].unique()

# Словари отображения
user_id_map = {uid: i for i, uid in enumerate(user_ids)}
item_id_map = {iid: i for i, iid in enumerate(item_ids)}

# Преобразование в индексы
purchases["user_idx"] = purchases["visitorid"].map(user_id_map)
purchases["item_idx"] = purchases["itemid"].map(item_id_map)

# Создание sparse-матрицы взаимодействий
n_users = len(user_id_map)
n_items = len(item_id_map)
interactions = sp.coo_matrix(
    (np.ones(len(purchases)), (purchases["user_idx"], purchases["item_idx"])),
    shape=(n_users, n_items)
)

# Обучение модели LightFM c лучшими параметрами
best_params = {
    'user_alpha': 1e-05,
    'no_components': 64,
    'loss': 'warp',
    'learning_rate': 0.1,
    'item_alpha': 1e-05,
    'random_state': 42,
}
# model = LightFM(loss='warp', no_components=30, random_state=42)
model = LightFM(**best_params)
model.fit(interactions, epochs=10, num_threads=4)

# Сохранение модели и отображений
os.makedirs("models", exist_ok=True)
joblib.dump((model, user_id_map, item_id_map), MODEL_PATH)
print("LightFM model trained and saved to:", MODEL_PATH)

