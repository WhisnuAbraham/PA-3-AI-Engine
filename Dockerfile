# Use official Python image (full version to ensure all C/C++ libraries exist for PyTorch)
FROM python:3.10

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

# Install semua requirements (tanpa torch dulu)
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch CPU version TERAKHIR dengan force-reinstall
# agar tidak bisa ditimpa oleh dependensi transformers dari PyPI
RUN pip install --no-cache-dir --force-reinstall torch --index-url https://download.pytorch.org/whl/cpu

# Verifikasi torch dan transformers berjalan dengan benar sebelum app di-deploy
RUN python -c "import torch; print('PyTorch OK:', torch.__version__); from transformers import BertForSequenceClassification; print('Transformers OK')"

# Copy the rest of the application files
COPY --chown=user:user . .

# Switch to the non-root user
USER user

# Expose the required Hugging Face Spaces port
EXPOSE 7860

# Run FastAPI app with uvicorn on port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
