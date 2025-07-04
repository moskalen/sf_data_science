import pandas as pd
import joblib

# Пути
DATA_PATH = "data/combined_features.csv"
MODEL_PATH = "models/catboost_model.pkl"
OUTPUT_PATH = "models/recommendations_catboost_top3.csv"

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

# Признаки и целевая переменная
drop_cols = [
    "visitorid", "itemid",
    "user_item_purchase_count",
    "last_interaction", "last_property_update"
]
X = df.drop(columns=drop_cols, errors="ignore")

# Преобразуем категориальные признаки в строку
cat_features = ["item_category_id", "parentid", "category_level"]
for col in cat_features:
    X[col] = X[col].astype(str)

# Предсказание вероятности покупки
df["purchase_proba"] = model.predict_proba(X)[:, 1]

# Получение топ-3 рекомендаций на пользователя
top3 = (
    df.sort_values(by=["visitorid", "purchase_proba"], ascending=[True, False])
    .groupby("visitorid")
    .head(3)[["visitorid", "itemid", "purchase_proba"]]
)

# Сохраняем
top3.to_csv(OUTPUT_PATH, index=False)
print("Top-3 recommendations saved to:", OUTPUT_PATH)
