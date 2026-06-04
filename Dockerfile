# Use official lightweight Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    HOME=/home/user

# Create a non-root user for security and Hugging Face compatibility
RUN useradd -m -u 1000 user
WORKDIR $HOME/app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY --chown=user:user requirements.txt .

# Install PyTorch CPU version explicitly (saves ~2GB of space and RAM)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install other python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY --chown=user:user . .

# Switch to the non-root user
USER user

# Expose the required Hugging Face Spaces port
EXPOSE 7860

# Run FastAPI app with uvicorn on port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
