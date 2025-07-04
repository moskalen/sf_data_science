import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
import joblib

# Пути
MODEL_PATH = "models/random_forest_combined.pkl"
DATA_PATH = "data/combined_features.csv"

if os.path.exists(MODEL_PATH):
    print("Model already exists. Skipping training.")
    exit()

# Загрузка данных
df = pd.read_csv(DATA_PATH)

# Целевая переменная
df["target"] = df["user_item_purchase_count"].apply(lambda x: 1 if x > 0 else 0)

# Балансировка классов
positive = df[df["target"] == 1]
negative = df[df["target"] == 0].sample(n=min(len(positive) * 5, len(df[df["target"] == 0])), random_state=42)
balanced_df = pd.concat([positive, negative]).sample(frac=1, random_state=42)

# Признаки
X = balanced_df.drop(columns=[
    "visitorid", "itemid", "user_item_purchase_count", "last_interaction", "last_property_update", "target"
], errors="ignore")

y = balanced_df["target"]

# Проверка на бесконечности и NaN
if not np.isfinite(X.to_numpy()).all():
    print("Найдены бесконечные или NaN значения. Очистим.")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

# Обучение модели
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X, y)

# Сохранение модели
os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_PATH)
print("✅ Model trained and saved to models/random_forest_combined.pkl")
