import requests
import sys
import os

# Добавляем текущий рабочий каталог в sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app')))

# Теперь можно импортировать модуль app
from utils import get_random_data


# Генерация случайных данных
df = get_random_data(1)
data = df.to_dict(orient='records')[0]

# Отправка POST-запроса на сервер для получения предсказания
response = requests.post('http://localhost:5000/predict', json=data)

# Печать ответа от сервера
print(response.json())
