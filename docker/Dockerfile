# Example Dockerfile (starting point)
FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip && pip install -r requirements.txt
CMD ["python", "scripts/run_camera.py", "--config", "config.yaml"]
