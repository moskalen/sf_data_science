import pandas as pd
from collections import defaultdict

# Пути к данным
RECS_PATH = "models/recommendations_catboost_top3.csv"
EVENTS_TEST_PATH = "data/events_test.csv"

# Загрузка
recommendations = pd.read_csv(RECS_PATH)
events_test = pd.read_csv(EVENTS_TEST_PATH)

# Выделим события покупки
purchases = events_test[events_test["event"] == "transaction"]

# Создаем словарь {visitorid: set(itemid)} для покупателей
purchased_dict = defaultdict(set)
for _, row in purchases.iterrows():
    purchased_dict[row["visitorid"]].add(row["itemid"])

# Группируем рекомендации по visitorid
grouped_recs = recommendations.groupby("visitorid")["itemid"].apply(list)

# Precision@3: только для пользователей с покупками
precision_sum = 0
num_users_with_purchases = 0

# Precision@3: глобальный по всем пользователям
global_hits = 0
total_recs = 0

for visitor_id, recs in grouped_recs.items():
    actual_purchases = purchased_dict.get(visitor_id, set())

    # Покупатели
    if actual_purchases:
        hits = sum(1 for item in recs[:3] if item in actual_purchases)
        precision_sum += hits / 3
        num_users_with_purchases += 1

    # Глобально
    hits = sum(1 for item in recs[:3] if item in actual_purchases)
    global_hits += hits
    total_recs += 3

# Вывод результатов
precision_purchasers = precision_sum / num_users_with_purchases if num_users_with_purchases else 0
precision_global = global_hits / total_recs if total_recs else 0

print(f"Precision@3 (Catboost): {precision_purchasers:.4f}")
print(f"Precision@3 (Catboost Global):   {precision_global:.4f}")

# Precision@3 (Catboost): 0.4128
# Precision@3 (Catboost Global):   0.0007