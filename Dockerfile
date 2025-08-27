FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgtk-3-0 \
    ffmpeg \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create necessary directories
RUN mkdir -p static/videos static/clips Channel_analysis/outputs/channels

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD wget -qO- http://localhost:8501/_stcore/health || exit 1

# Run the app
CMD ["streamlit", "run", "youtube_strategy_app.py", "--server.port=8501", "--server.address=0.0.0.0"]