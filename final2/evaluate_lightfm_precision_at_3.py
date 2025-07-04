import pandas as pd
from collections import defaultdict

# Пути к файлам
RECS_PATH = "models/recommendations_lightfm_top3.csv"
EVENTS_TEST_PATH = "data/events_test.csv"

# Загрузка рекомендаций
recommendations = pd.read_csv(RECS_PATH)

# Загрузка тестовых событий
events_test = pd.read_csv(EVENTS_TEST_PATH)

# Фильтрация покупок
purchases = events_test[events_test["event"] == "transaction"]

# Сопоставление: visitorid -> set(itemid)
purchased_dict = defaultdict(set)
for _, row in purchases.iterrows():
    purchased_dict[row["visitorid"]].add(row["itemid"])

# Группируем рекомендации по visitorid
grouped_recs = recommendations.groupby("visitorid")["itemid"].apply(list)

# Precision@3
precision_sum = 0
user_count = 0

for visitorid, recommended_items in grouped_recs.items():
    actual_purchases = purchased_dict.get(visitorid, set())
    if not actual_purchases:
        continue
    hits = sum(1 for item in recommended_items[:3] if item in actual_purchases)
    precision_sum += hits / 3
    user_count += 1

precision_at_3 = precision_sum / user_count if user_count > 0 else 0.0

print(f"Precision@3 (LightFM): {precision_at_3:.4f}")

# Precision@3 (LightFM): 0.2727


# Список покупок в тесте
purchases = events_test[events_test["event"] == "transaction"][["visitorid", "itemid"]]

# Топ-3 рекомендации для каждого пользователя
top3 = (
    recommendations.sort_values(by=["visitorid", "score"], ascending=[True, False])
    .groupby("visitorid")
    .head(3)[["visitorid", "itemid"]]
)

# Precision@3
merged = top3.merge(purchases, on=["visitorid", "itemid"], how="left", indicator=True)
precision_at_3 = (merged["_merge"] == "both").sum() / top3.shape[0]

print(f"Precision@3 (LightFM Global): {precision_at_3:.4f}")
