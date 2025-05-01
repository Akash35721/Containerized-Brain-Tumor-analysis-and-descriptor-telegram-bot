# Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables (avoids Python buffering output, sets workdir)
ENV PYTHONUNBUFFERED=1
ENV WORKDIR=/app

# Set the working directory in the container
WORKDIR ${WORKDIR}

# Install any needed system dependencies if required by ultralytics/torch
# (Often needed for OpenCV dependencies. Uncomment if you encounter issues)
# RUN apt-get update && apt-get install -y --no-install-recommends libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Install system dependencies needed by OpenCV (cv2), which is used by ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    # Clean up apt cache to reduce image size
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create a directory for models inside the container
# Note: We copy the model file itself via docker-compose volumes usually,
# but creating the directory here is fine.
RUN mkdir -p ${WORKDIR}/models

# Copy your application code into the container
COPY app.py .

# Command to run your application when the container starts
# The actual environment variables (like token, paths) will be passed via docker-compose
CMD ["python", "app.py"]



