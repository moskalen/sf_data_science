FROM python:3.9-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл requirements.txt в рабочую директорию
COPY requirements.txt requirements.txt

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем остальные файлы приложения в рабочую директорию
COPY . .

# Открываем порт 5000 для Flask
EXPOSE 5000

# Команда для запуска приложения
CMD ["python", "app/server.py"]
