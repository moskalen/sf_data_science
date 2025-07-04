import pandas as pd
import joblib
import numpy as np

# Пути
model_path = "models/knn_model.pkl"
features_path = "data/user_item_features.csv"
output_path = "models/recommendations_knn_top3.csv"

# Загрузка
knn = joblib.load(model_path)
df = pd.read_csv(features_path)

# Подготовка данных
X = df.drop(columns=["visitorid", "itemid", "last_interaction"], errors="ignore")
df["score"] = 0.0

# Предсказание: средняя дистанция до ближайших соседей (меньше — лучше)
distances, _ = knn.kneighbors(X)
df["score"] = -distances.mean(axis=1)  # отрицательная дистанция как прокси для "похожести"

# Топ-3 товара на пользователя
top3 = (
    df.sort_values(["visitorid", "score"], ascending=[True, False])
    .groupby("visitorid")
    .head(3)
    .reset_index(drop=True)
)

# Сохранение
top3[["visitorid", "itemid", "score"]].to_csv(output_path, index=False)
print(f"🎯 KNN рекомендации сохранены в {output_path}")
