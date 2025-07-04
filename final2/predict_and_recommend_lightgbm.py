import pandas as pd
import joblib
import os

# Пути
MODEL_PATH = "models/lightgbm_model.pkl"
DATA_PATH = "data/combined_features.csv"
OUTPUT_PATH = "models/recommendations_lightgbm_top3.csv"

# Загрузка данных
df = pd.read_csv(DATA_PATH)

# Удаляем строки с пропущенными значениями
df = df.dropna()

# Преобразуем типы признаков, если нужно
numeric_columns = [
    'recency', 'user_item_interaction_share', 'item_user_interaction_share',
    'item_purchase_rate', 'item_add_to_cart_rate',
    'item_category_id', 'parentid', 'category_level', 'unique_properties_count'
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Удаляем строки с NaN после преобразования типов
df = df.dropna()

# Целевые переменные, которые нельзя подавать в модель
X = df.drop(columns=[
    "visitorid", "itemid", "user_item_purchase_count", "last_interaction", "last_property_update"
], errors="ignore")

# Загрузка модели
model = joblib.load(MODEL_PATH)

# Предсказание вероятностей покупки
df["purchase_proba"] = model.predict_proba(X)[:, 1]

# Выбор top-3 рекомендаций для каждого пользователя
top3 = (
    df.sort_values(["visitorid", "purchase_proba"], ascending=[True, False])
    .groupby("visitorid")
    .head(3)
)

# Сохраняем результат
os.makedirs("models", exist_ok=True)
top3.to_csv(OUTPUT_PATH, index=False)

print(f"Recommendations saved to {OUTPUT_PATH}")
