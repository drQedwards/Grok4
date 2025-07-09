FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install Grok4 package
RUN pip install -e .

# Expose port for API server
EXPOSE 8000

# Set default command
CMD ["grok4", "serve", "--host", "0.0.0.0", "--port", "8000"]