FROM python:3.10-slim

# Установка зависимостей системы
RUN apt-get update && apt-get install -y build-essential

# Создание рабочей директории
WORKDIR /app

# Копирование необходимых файлов
COPY app.py /app/
COPY prepare.py /app/
COPY requirements.txt /app/
COPY models/lightfm_model_best.pkl /app/models/
COPY models/interaction_matrix.npz /app/models/

# Установка зависимостей
RUN pip install --upgrade pip

# Установка Python-зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Команда запуска FastAPI с Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
