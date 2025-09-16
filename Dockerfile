FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY wellness_agent_whatsapp_json_redis.py ./
EXPOSE 8000
CMD ["uvicorn", "wellness_agent_whatsapp_json_redis:app", "--host", "0.0.0.0", "--port", "8000"]
