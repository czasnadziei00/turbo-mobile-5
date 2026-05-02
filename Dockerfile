FROM python:3.12-slim

# Install system dependencies for PaddleOCR + OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    libgomp1 \
    libstdc++6 \
    libjpeg-dev \
    zlib1g \
    libpng-dev \
    libtiff5 \
    libopenblas-dev \
    libatlas-base-dev \
    liblapack-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
