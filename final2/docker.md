# Инструкция по сборке и запуску контейнера

## Загрузить Docker-image:

* [скачать архив с Docker-image](https://drive.google.com/file/d/1kvMJRTySrqvTzbzPJK2OO-suQsG6RAb5/view?usp=sharing)
* раскрыть архив:
    * Linux/Mac:
        ```bash
        gzip -d recommender_image.tar.gz
        ```
    * Windows (с помощью 7-Zip):
        ```
        Клик ПКМ → 7-Zip → "Extract Here" или "Open archive"
        ```
* загрузить Docker-image:
    ```bash
    docker load -i recommender_image.tar
    ```

## Подготовка (если нет Docker-image)
В директории `models/` должны присутствовать следующие файлы:
- `lightfm_model_best.pkl` — обученная модель LightFM
- `interaction_matrix.npz` — матрица взаимодействий (созданная в `prepare.py`)

## Запуск контейнеров
```bash
docker-compose up --build
```

## Сервисы:

* Приложение: http://localhost:8000/docs

* Prometheus: http://localhost:9090

* Grafana: http://localhost:3000 (логин/пароль по умолчанию: `admin` / `admin`)

## В Grafana:

Установите в Graphana (`Home -> Connections -> Data Soource`) новый источник данных Prometheus:

```http://prometheus:9090```


## Пример запросов

Все запросы можно делать из браузера в http://localhost:8000/docs

<img src=images/test_service.png>


#### Поолучить рекоммендации для пользователя
```
GET http://localhost:8000/visitors/670282/recommendations
```

#### Пример ответа:
```
{
  "visitor_id": 670282,
  "recommendations": [106547, 272152, 316753]
}
```

#### Пример запроса метрики
```
GET http://localhost:8000/metrics
```

#### Пример ответа:
```
# HELP python_gc_objects_collected_total Objects collected during gc
# TYPE python_gc_objects_collected_total counter
python_gc_objects_collected_total{generation="0"} 597.0
python_gc_objects_collected_total{generation="1"} 166.0
python_gc_objects_collected_total{generation="2"} 0.0
# HELP python_gc_objects_uncollectable_total Uncollectable objects found during GC
# TYPE python_gc_objects_uncollectable_total counter
python_gc_objects_uncollectable_total{generation="0"} 0.0
python_gc_objects_uncollectable_total{generation="1"} 0.0
python_gc_objects_uncollectable_total{generation="2"} 0.0
# HELP python_gc_collections_total Number of times this generation was collected
# TYPE python_gc_collections_total counter
python_gc_collections_total{generation="0"} 219.0
python_gc_collections_total{generation="1"} 19.0
python_gc_collections_total{generation="2"} 1.0
# HELP python_info Python platform information
# TYPE python_info gauge
python_info{implementation="CPython",major="3",minor="10",patchlevel="18",version="3.10.18"} 1.0
# HELP process_virtual_memory_bytes Virtual memory size in bytes.
# TYPE process_virtual_memory_bytes gauge
process_virtual_memory_bytes 3.68111616e+08
# HELP process_resident_memory_bytes Resident memory size in bytes.
# TYPE process_resident_memory_bytes gauge
process_resident_memory_bytes 1.04251392e+08
# HELP process_start_time_seconds Start time of the process since unix epoch in seconds.
# TYPE process_start_time_seconds gauge
process_start_time_seconds 1.75173559581e+09
# HELP process_cpu_seconds_total Total user and system CPU time spent in seconds.
# TYPE process_cpu_seconds_total counter
process_cpu_seconds_total 4.7
# HELP process_open_fds Number of open file descriptors.
# TYPE process_open_fds gauge
process_open_fds 10.0
# HELP process_max_fds Maximum number of open file descriptors.
# TYPE process_max_fds gauge
process_max_fds 1.048576e+06
# HELP recommend_requests_total Total number of recommendation requests
# TYPE recommend_requests_total counter
recommend_requests_total 1.0
# HELP recommend_requests_created Total number of recommendation requests
# TYPE recommend_requests_created gauge
recommend_requests_created 1.7517355997285564e+09
# HELP recommendation_duration_seconds Recommendation response time
# TYPE recommendation_duration_seconds histogram
recommendation_duration_seconds_bucket{le="0.005"} 0.0
recommendation_duration_seconds_bucket{le="0.01"} 0.0
recommendation_duration_seconds_bucket{le="0.025"} 0.0
recommendation_duration_seconds_bucket{le="0.05"} 0.0
recommendation_duration_seconds_bucket{le="0.075"} 0.0
recommendation_duration_seconds_bucket{le="0.1"} 0.0
recommendation_duration_seconds_bucket{le="0.25"} 0.0
recommendation_duration_seconds_bucket{le="0.5"} 0.0
recommendation_duration_seconds_bucket{le="0.75"} 0.0
recommendation_duration_seconds_bucket{le="1.0"} 0.0
recommendation_duration_seconds_bucket{le="2.5"} 0.0
recommendation_duration_seconds_bucket{le="5.0"} 0.0
recommendation_duration_seconds_bucket{le="7.5"} 0.0
recommendation_duration_seconds_bucket{le="10.0"} 0.0
recommendation_duration_seconds_bucket{le="+Inf"} 0.0
recommendation_duration_seconds_count 0.0
recommendation_duration_seconds_sum 0.0
# HELP recommendation_duration_seconds_created Recommendation response time
# TYPE recommendation_duration_seconds_created gauge
recommendation_duration_seconds_created 1.7517355997286167e+09
```