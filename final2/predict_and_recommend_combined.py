import pandas as pd
import numpy as np
import joblib

# Пути
MODEL_PATH = "models/random_forest_combined.pkl"
DATA_PATH = "data/combined_features.csv"
OUTPUT_PATH = "models/recommendations_random_forest_top3.csv"

# Загрузка модели
model = joblib.load(MODEL_PATH)

# Загрузка признаков
df = pd.read_csv(DATA_PATH)

# Удаляем строки с бесконечностями или NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Сохраняем visitorid и itemid
meta = df[["visitorid", "itemid"]].copy()

# Выделяем признаки
X = df.drop(columns=[
    "visitorid", "itemid", "user_item_purchase_count",
    "last_interaction", "last_property_update"
], errors="ignore")

# Предсказание вероятности покупки
df["purchase_proba"] = model.predict_proba(X)[:, 1]

# Соединяем с meta для группировки по пользователю
df[["visitorid", "itemid", "purchase_proba"]].to_csv("full_scored.csv", index=False)

# Получаем топ-3 рекомендации на пользователя
top_recommendations = (
    df.sort_values(["visitorid", "purchase_proba"], ascending=[True, False])
      .groupby("visitorid")
      .head(3)
      .reset_index(drop=True)
)

# Сохраняем результат
top_recommendations.to_csv(OUTPUT_PATH, index=False)
print(f"Top-3 recommendations saved to: {OUTPUT_PATH}")
