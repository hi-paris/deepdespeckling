# Use official Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies including GDAL
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL config paths so pip can find it
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Pre-install GDAL with pip using system version
RUN pip install --no-cache-dir "gdal==3.6.2"

# Clone deepdespeckling repo
RUN git clone https://github.com/hi-paris/deepdespeckling.git /app/deepdespeckling

# Install deepdespeckling dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Install deepdespeckling as a module
RUN pip install -e /app/deepdespeckling

# Install JupyterLab
RUN pip install --no-cache-dir jupyterlab

# Expose JupyterLab port
EXPOSE 8888

# Default command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
