FROM python:3.10.13-slim-bullseye

# Install build dependencies
RUN apt-get update && \
    apt-get install -y build-essential && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
COPY train.py predct.py test_predict.ipynb Dockerfile Capstone.ipynb ping.py salaries.csv ./
RUN pip install --no-cache-dir -r /app/requirements.txt

# Set the working directory to /app
WORKDIR /app

# Copy the necessary files to the container at /app
COPY . /app










