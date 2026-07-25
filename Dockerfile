# AI Medical Intelligence Platform — Docker image
# Build:  docker build -t ai-medical-platform .
# Run:    docker run -p 8000:8000 --env-file .env -v $(pwd)/models:/app/models ai-medical-platform

FROM python:3.11-slim

# System dependencies required by opencv-python-headless and matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The trained model is NOT baked into the image (it's large and changes
# independently of code). Mount it as a volume, or copy models/ in before
# building if you want a fully self-contained image.
RUN mkdir -p /app/models /app/reports/gradcam_outputs

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
