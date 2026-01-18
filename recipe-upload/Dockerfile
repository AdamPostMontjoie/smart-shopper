FROM python:3.12-slim

WORKDIR /app

# 1. Clean up apt list to save space
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# 2. CRITICAL: Install CPU-only PyTorch first (Saves ~1.5GB)
# We use --no-cache-dir to stop pip from saving the download file in the image
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 3. Install the rest of your stack
RUN pip install --no-cache-dir pandas supabase python-dotenv datasets sentence-transformers

COPY upload.py .
COPY .env . 

CMD ["python", "upload.py"]