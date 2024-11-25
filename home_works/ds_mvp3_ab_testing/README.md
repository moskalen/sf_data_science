# Домашнее задание: анализ A/A/B теста

Есть A/A/B-тестирования от одного известного маркетплейса.

> sample_a, sample_c — АА-группы, sample_b — отдельная группа.

В каждом датасете есть три типа действий пользователей: 0 — клик, 1 — просмотр и 2 — покупка (пользователь просматривает выдачу товаров, кликает на понравившийся товар и совершает покупку).

Маркетплейс ориентируется на следующие метрики:

* ctr (отношение кликов к просмотрам товаров);
* purchase rate (отношение покупок к просмотрам товаров);
* gmv (оборот, сумма произведений количества покупок на стоимость покупки), где считаем 1 сессию за 1 точку (1 сессия на 1 пользователя).

### Задача — понять, нет ли проблемы с разъезжанием сплитов и улучшает ли алгоритм B работу маркетплейса.

## Инструкция для проверки проекта
1. Загрузите ноутбук [`main.ipynb`](main.ipynb)

2. Загрузите [архив с данными](https://drive.google.com/file/d/1W2MFjbOW2dIdOrrSHKxPtnMz8OznnfBa/view?usp=sharing) и скопируйте его в ту же папку, что и `main.ipynb` ноутбук под именем `data.zip`.

3. Раскройте архив `data.zip`. Должна появиться папка `data/` со следующими файлами:
   * `sample_a.zip`
   * `sample_b.zip`
   * `sample_c.zip`
   * `item_prices.zip`

3. Загрузите [`requirements.txt`](requirements.txt)
