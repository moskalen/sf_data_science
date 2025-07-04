import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Загрузка events_train
events_train = pd.read_csv("data/events_train.csv")

# Оставляем только взаимодействия, подходящие под implicit feedback
events_filtered = events_train[events_train["event"].isin(["view", "addtocart", "transaction"])]

# Строим словари для индексов
user_ids = events_filtered["visitorid"].unique()
item_ids = events_filtered["itemid"].unique()

user_id_map = {uid: idx for idx, uid in enumerate(user_ids)}
item_id_map = {iid: idx for idx, iid in enumerate(item_ids)}

# Обратные отображения
reverse_user_id_map = {idx: uid for uid, idx in user_id_map.items()}
reverse_item_id_map = {idx: iid for iid, idx in item_id_map.items()}

# Матрица взаимодействий: user x item
rows = events_filtered["visitorid"].map(user_id_map)
cols = events_filtered["itemid"].map(item_id_map)
data = np.ones(len(events_filtered))

interaction_matrix = coo_matrix((data, (rows, cols)), shape=(len(user_ids), len(item_ids))).tocsr()

# Обучаем KNN по item-item матрице
knn_model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=20)
knn_model.fit(interaction_matrix.T)  # Transposed: item-item

# Получаем пользователей из test
events_test = pd.read_csv("data/events_test.csv")
test_users = events_test["visitorid"].unique()

recommendations = []

for uid in test_users:
    if uid not in user_id_map:
        continue

    uidx = user_id_map[uid]
    user_row = interaction_matrix[uidx]
    interacted_items = user_row.indices

    scores = {}
    for iid in interacted_items:
        dists, neighbors = knn_model.kneighbors(interaction_matrix.T[iid], return_distance=True)
        for neighbor_idx, dist in zip(neighbors[0], dists[0]):
            if neighbor_idx not in interacted_items:
                scores[neighbor_idx] = scores.get(neighbor_idx, 0) + (1 - dist)

    top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]

    for item_idx, score in top_items:
        recommendations.append({
            "visitorid": uid,
            "itemid": reverse_item_id_map[item_idx],
            "score": score
        })

# Сохраняем рекомендации
df_recs = pd.DataFrame(recommendations)
df_recs.to_csv("models/recommendations_knn_cf_top3.csv", index=False)
print("Рекомендации сохранены в models/recommendations_knn_cf_top3.csv")
