import pandas as pd
import joblib
import os
from catboost import CatBoostClassifier

# Пути
DATA_PATH = "data/combined_features.csv"
MODEL_PATH = "models/catboost_model.pkl"

# Пропуск, если модель уже обучена
if os.path.exists(MODEL_PATH):
    print("CatBoost model already exists. Skipping training.")
    exit()

# Загрузка
df = pd.read_csv(DATA_PATH)

# Целевая переменная
df["target"] = df["user_item_purchase_count"].apply(lambda x: 1 if x > 0 else 0)

# Предположим, вы считаете эти признаки категориальными:
categorical_cols = ["item_category_id", "parentid", "category_level"]

# Преобразуем их в строки
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype(str)

# Балансировка
positive = df[df["target"] == 1]
negative = df[df["target"] == 0].sample(n=len(positive) * 5, random_state=42)
balanced_df = pd.concat([positive, negative]).sample(frac=1, random_state=42)

# Выделяем категориальные признаки (если они есть, например: item_category_id, parentid и т.п.)
cat_features = ["item_category_id", "parentid", "category_level"]  # только если они категориальные по смыслу

# Признаки и целевая
drop_cols = ["visitorid", "itemid", "user_item_purchase_count", "last_interaction", "last_property_update", "target"]
X = balanced_df.drop(columns=drop_cols)
y = balanced_df["target"]

best_params = {
    'iterations': 457,
    'depth': 4,
    'learning_rate': 0.2210108434501312,
    'l2_leaf_reg': 0.009961523122557155,
    'random_strength': 0.017734174437101843,
    'border_count': 145,
    'random_seed': 42,
    'verbose': 50,
    'cat_features': cat_features
}
model = CatBoostClassifier(**best_params)

# Обучение
model.fit(X, y)

# Сохранение
os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_PATH)
print("CatBoost model saved to:", MODEL_PATH)

# 0:	learn: 0.5060762	total: 88.7ms	remaining: 26.5s
# 50:	learn: 0.0056518	total: 1.52s	remaining: 7.42s
# 100:	learn: 0.0045878	total: 2.77s	remaining: 5.45s
# 150:	learn: 0.0040009	total: 4s	remaining: 3.95s
# 200:	learn: 0.0037451	total: 5.16s	remaining: 2.54s
# 250:	learn: 0.0034864	total: 6.35s	remaining: 1.24s
# 299:	learn: 0.0032618	total: 7.47s	remaining: 0us
# CatBoost model saved to: models/catboost_model.pkl
