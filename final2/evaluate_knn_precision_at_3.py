import pandas as pd
from collections import defaultdict

# Пути к файлам
RECS_PATH = "models/recommendations_knn_top3.csv"
EVENTS_TEST_PATH = "data/events_test.csv"

# Загрузка данных
recommendations = pd.read_csv(RECS_PATH)
events_test = pd.read_csv(EVENTS_TEST_PATH)

# Фильтрация событий на покупки
purchases_test = events_test[events_test["event"] == "transaction"]

# Сопоставим (visitorid -> set of purchased itemid)
purchased_dict = defaultdict(set)
for _, row in purchases_test.iterrows():
    purchased_dict[row["visitorid"]].add(row["itemid"])

# Группируем рекомендации по visitorid
recommendations_grouped = recommendations.groupby("visitorid")["itemid"].apply(list)

# Подсчёт Precision@3
precision_sum = 0
num_users_with_recs = 0

for visitor_id, recs in recommendations_grouped.items():
    actual_purchases = purchased_dict.get(visitor_id, set())
    if not actual_purchases:
        continue  # У пользователя не было покупок — пропускаем
    hits = sum(1 for item in recs[:3] if item in actual_purchases)
    precision = hits / 3
    precision_sum += precision
    num_users_with_recs += 1

precision_at_3 = precision_sum / num_users_with_recs if num_users_with_recs > 0 else 0.0

print(f"Precision@3 (Random Forest): {precision_at_3:.4f}")

