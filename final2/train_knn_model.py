import pandas as pd
from sklearn.neighbors import NearestNeighbors
import joblib
import os

# Пути
data_path = "data/user_item_features.csv"
model_path = "models/knn_model.pkl"

# Загрузка данных
df = pd.read_csv(data_path)

# Отбор признаков
X = df.drop(columns=["visitorid", "itemid", "last_interaction"], errors="ignore")

# Обучение KNN модели
knn = NearestNeighbors(n_neighbors=3, metric='cosine')
knn.fit(X)

# Сохраняем модель
os.makedirs("models", exist_ok=True)
joblib.dump(knn, model_path)
print("KNN модель сохранена в models/knn_model.pkl")
