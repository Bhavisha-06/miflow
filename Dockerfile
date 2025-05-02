FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy MiFlow package
COPY . .

# Install the package
RUN pip install -e .

# Create directory for videos
RUN mkdir -p /data

# Set working directory to /data
WORKDIR /data

# Set the entrypoint
ENTRYPOINT ["miflow"]

# Default command (can be overridden)
CMD ["--help"]
