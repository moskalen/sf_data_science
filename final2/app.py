from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel
from starlette.responses import Response
from typing import List
import joblib
import numpy as np
import uvicorn

app = FastAPI()

# Метрики Prometheus
REQUEST_COUNT = Counter("recommend_requests_total", "Total number of recommendation requests")
RECOMMENDATION_DURATION = Histogram("recommendation_duration_seconds", "Recommendation response time")

# Загрузка модели и вспомогательных структур
model, user_id_map, item_id_map = joblib.load("models/lightfm_model_best.pkl")
interaction_matrix = joblib.load("models/interaction_matrix.npz")

# Инвертированные маппинги
user_idx_map = {v: k for k, v in user_id_map.items()}
item_idx_map = {v: k for k, v in item_id_map.items()}


class RecommendationsResponse(BaseModel):
    visitor_id: int
    recommendations: List[int]


@RECOMMENDATION_DURATION.time()
@app.get("/visitors/{visitor_id}/recommendations", response_model=RecommendationsResponse)
async def recommend(visitor_id: int):
    REQUEST_COUNT.inc()

    if visitor_id not in user_id_map:
        raise HTTPException(status_code=404, detail="Пользователь не найден в обучающем наборе")

    user_idx = user_id_map[visitor_id]

    scores = model.predict(user_ids=user_idx,
                           item_ids=np.arange(len(item_id_map)))

    known_items = set(interaction_matrix.getrow(user_idx).indices)
    item_scores = [(item, score) for item, score in enumerate(scores) if item not in known_items]

    top_items = sorted(item_scores, key=lambda x: x[1], reverse=True)[:3]
    top_item_ids = [item_idx_map[i] for i, _ in top_items]

    return {"visitor_id": visitor_id, "recommendations": top_item_ids}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Для локального запуска
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
