FROM python:3.11-slim as builder

WORKDIR /app

# Копируем зависимости для кэширования слоя
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt


FROM python:3.11-slim

WORKDIR /app

# Создаем non-root пользователя для безопасности
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Копируем установленные зависимости из builder stage
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Копируем исходный код приложения
COPY src/ ./src/

# Health check для мониторинга состояния сервиса
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Точка входа по умолчанию - Search API
CMD ["python", "src/search/main.py"]

# Для запуска ETL сервиса используйте: 
# docker run --rm vectordb python src/etl/main.py