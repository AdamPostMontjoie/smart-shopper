FROM python:3.12-slim

WORKDIR /app

# 1. Install system dependencies
# We include 'gnupg' and 'curl'/'wget' which are required for the chrome setup
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    wget \
    gnupg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 2. INSTALL GOOGLE CHROME (Fixed for Debian 12 / Python 3.12)
# We download the key, convert it to a gpg keyring, and point apt to it safely
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | \
    gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main" \
    > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# 3. Install CPU-only Torch (Saves Space)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 4. Install Requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy Code
COPY . .

CMD ["python", "basket.py"]