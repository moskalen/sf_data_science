import pandas as pd
import numpy as np
import joblib
from scipy.sparse import coo_matrix


def build_interaction_matrix(events, user_id_map, item_id_map):
    purchases = events[events["event"] == "transaction"]
    user_idx = purchases["visitorid"].map(user_id_map)
    item_idx = purchases["itemid"].map(item_id_map)
    data = np.ones(len(user_idx), dtype=np.int32)
    matrix = coo_matrix((data, (user_idx, item_idx)),
                        shape=(len(user_id_map), len(item_id_map)))
    return matrix


def run():
    events = pd.read_csv("data/events_train.csv")
    _, user_id_map, item_id_map = joblib.load("models/lightfm_model_best.pkl")

    interaction_matrix = build_interaction_matrix(events, user_id_map, item_id_map)

    joblib.dump(interaction_matrix, "models/interaction_matrix.npz")


if __name__ == "__main__":
    run()
