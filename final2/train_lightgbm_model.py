import pandas as pd
import lightgbm as lgb
import joblib
import os

# Пути
DATA_PATH = "data/combined_features.csv"
MODEL_PATH = "models/lightgbm_model.pkl"

# Пропускаем обучение, если модель уже существует
if os.path.exists(MODEL_PATH):
    print("LightGBM model already exists. Skipping training.")
    exit()

# Загрузка данных
df = pd.read_csv(DATA_PATH)

# Целевая переменная
df["target"] = df["user_item_purchase_count"].apply(lambda x: 1 if x > 0 else 0)

# Балансировка классов
positive = df[df["target"] == 1]
negative = df[df["target"] == 0].sample(n=len(positive) * 5, random_state=42)
balanced_df = pd.concat([positive, negative]).sample(frac=1, random_state=42)

# Признаки
drop_cols = [
    "visitorid", "itemid",
    "user_item_purchase_count",
    "last_interaction", "last_property_update",
    "target"
]
X = balanced_df.drop(columns=drop_cols, errors="ignore")
y = balanced_df["target"]

# Обучение
model = lgb.LGBMClassifier(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.05,
    random_state=42,
    class_weight="balanced"
)
model.fit(X, y)

# Сохраняем
os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_PATH)
print("LightGBM model trained and saved to:", MODEL_PATH)
