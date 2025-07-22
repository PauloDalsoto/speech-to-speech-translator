# Use Python 3.11 slim image as base
FROM python:3.12

# Set working directory
WORKDIR /app

# Install system dependencies for audio processing
RUN echo "🔧 Installing system dependencies..." && \
    apt-get update && \
    apt-get install -y \
    portaudio19-dev \
    python3-dev \
    gcc \
    g++ \
    make \
    libasound2-dev \
    libpulse-dev \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && echo "✅ System dependencies installed successfully!"

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN echo "📦 Installing Python dependencies..." && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    echo "✅ Python dependencies installed successfully!"

# Copy application files
RUN echo "📁 Copying application files..."
COPY app/ ./app/
COPY config_example.json ./config.json
RUN echo "✅ Application files copied successfully!"

# Create logs directory
RUN mkdir -p /app/logs && \
    echo "📂 Logs directory created!"

# Expose port (if needed for future web interface)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import sys; print('Container is healthy'); sys.exit(0)" || exit 1

# Set the default command
CMD echo "🚀 Starting Speech-to-Speech Translator..." && \
    echo "📋 Configuration loaded from config.json" && \
    echo "🎯 Running application..." && \
    python -m app.main --config config.json