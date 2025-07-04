import pandas as pd
from collections import defaultdict

# Пути к файлам
RECS_PATH = "models/recommendations_knn_cf_top3.csv"
EVENTS_TEST_PATH = "data/events_test.csv"

# Загрузка данных
recommendations = pd.read_csv(RECS_PATH)
events_test = pd.read_csv(EVENTS_TEST_PATH)

# Отбираем покупки (transaction) из теста
purchases_test = events_test[events_test["event"] == "transaction"]

# Сопоставим: visitorid -> set of purchased itemid
purchased_dict = defaultdict(set)
for _, row in purchases_test.iterrows():
    purchased_dict[row["visitorid"]].add(row["itemid"])

# Группировка рекомендаций по пользователю
recommendations_grouped = recommendations.groupby("visitorid")["itemid"].apply(list)

# Подсчёт Precision@3 по покупателям
precision_sum = 0
users_with_purchases = 0

for visitor_id, recs in recommendations_grouped.items():
    actual_purchases = purchased_dict.get(visitor_id, set())
    if not actual_purchases:
        continue
    hits = sum(1 for item in recs[:3] if item in actual_purchases)
    precision_sum += hits / 3
    users_with_purchases += 1

precision_at_3 = precision_sum / users_with_purchases if users_with_purchases > 0 else 0.0
print(f"Precision@3 (KNN CF): {precision_at_3:.4f}")

# Подсчёт глобального Precision@3 (по всем пользователям)
global_hits = 0
total_recs = 0

for visitor_id, recs in recommendations_grouped.items():
    actual_purchases = purchased_dict.get(visitor_id, set())
    hits = sum(1 for item in recs[:3] if item in actual_purchases)
    global_hits += hits
    total_recs += 3

precision_global = global_hits / total_recs if total_recs > 0 else 0.0
print(f"Precision@3 (KNN CF Global): {precision_global:.4f}")

# Precision@3 (KNN CF): 0.0028
# Precision@3 (KNN CF Global): 0.0000
