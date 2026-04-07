FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080
ENV PYTHONUNBUFFERED=1
EXPOSE 8080

# 不依赖 run.py（避免仓库未包含该文件时容器启动失败）；Zeabur 会注入 PORT
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
