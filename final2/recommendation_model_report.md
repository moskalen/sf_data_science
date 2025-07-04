# Отчет по построению рекомендательной модели

## 1. Формат входных данных для обучения

Для обучения модели использовался объединённый датафрейм `combined_features.csv`, содержащий следующие признаки:

- `visitorid` (int64) — идентификатор пользователя;
- `itemid` (int64) — идентификатор товара;
- Поведенческие признаки:
  - `user_item_interactions`, `user_item_view_count`, `user_item_add_to_cart_count`, `user_item_purchase_count`;
  - `last_interaction`, `recency`;
  - `total_user_interactions`, `total_user_view_count`, `total_user_add_to_cart_count`, `total_user_purchase_count`;
  - `user_item_interaction_share`;
  - `total_item_interactions_x`, `total_item_view_count`, `total_item_add_to_cart_count`, `total_item_purchase_count`;
  - `unique_visitors`, `item_purchase_rate`, `item_add_to_cart_rate`, `item_user_interaction_share`;
- Категориальные признаки:
  - `item_category_id`, `parentid`, `category_level`, `unique_properties_count`, `last_property_update`.

Целевая переменная (`target`) создавалась на основе признака `user_item_purchase_count`: если значение больше 0 — целевая метка равна 1, иначе 0.

---

## 2. Трансформации исходного датасета

- Данные были сбалансированы: использованы все положительные примеры (`target == 1`) и случайная подвыборка отрицательных (`target == 0`) в соотношении 1:5.
- Некоторые категориальные признаки были преобразованы в строковый формат при обучении CatBoost.
- Пропущенные значения в категориальных признаках были заполнены строкой `"unknown"`.

---

## 3. Построение валидации

- Для оценки моделей применялась метрика **Precision@3**:
  - **Precision@3 (по покупателям)** — точность топ-3 рекомендаций среди пользователей, совершивших покупки;
  - **Precision@3 (глобальная)** — точность по всем пользователям, включая тех, кто ничего не купил.

---

## 4. Эксперименты

| Модель                  | Precision\@3 (по покупателям) | Precision\@3 (глобальная) |
| ----------------------- | ----------------------------- | ------------------------- |
| **Random Forest**       | **0.4050**                    | 0.0019                    |
| **CatBoost**            | **0.4128**                    | 0.0007                    |
| **LightGBM**            | 0.4058                        | 0.0019                    |
| **LightFM (best)**      | 0.2727                        | **0.3368**                |
| **KNN (на признаках)**  | 0.2777                        | 0.0010                    |
| **KNN (Collaborative)** | 0.0028                        | 0.0000                    |

- **Лучшие гиперпараметры для LightFM**:
  - `loss='warp'`, `no_components=64`, `learning_rate=0.1`, `user_alpha=1e-5`, `item_alpha=1e-5`.

- **CatBoost** тюнился с использованием Optuna, и в результате достиг наилучшего Precision@3 (по покупателям), но модель LightFM показала лучшие глобальные рекомендации.

---

## Вывод

С учётом задачи предоставления рекомендаций **всем пользователям**, была выбрана модель **LightFM** как финальная, несмотря на более высокое Precision@3 по покупателям у моделей на бустинге.
