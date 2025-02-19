# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgl1-mesa-glx \
    libgtk2.0-dev \
    libboost-python-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
FROM opencv/opencv:latest

# Manually install dlib and face_recognition if not included in requirements.txt
RUN pip install --no-cache-dir dlib==19.24.2 face_recognition opencv-python-headless

# Copy the application code into the container
COPY . .

# Create directories for uploads and outputs
RUN mkdir -p /app/uploads /app/outputs

# Expose the port the app runs on
EXPOSE 8000

# Set environment variables
ENV UPLOAD_FOLDER=/app/uploads
ENV OUTPUT_FOLDER=/app/outputs

# Command to run the FastAPI app with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

