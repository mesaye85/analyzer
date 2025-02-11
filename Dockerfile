FROM --platform=linux/amd64 python:3.10-slim


WORKDIR /app

# Install system dependencies and build tools
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    build-essential \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --only-binary=:all: -r requirements.txt && \
    python -m nltk.downloader punkt -d /usr/share/nltk_data

# Copy the rest of the application
COPY . .

CMD ["python", "main.py"]
