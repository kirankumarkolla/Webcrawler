FROM python:3.10-slim

# System dependencies for Playwright
RUN apt-get update && apt-get install -y \
    curl unzip git wget fonts-liberation libnss3 libxss1 libasound2 \
    libatk-bridge2.0-0 libgtk-3-0 libdrm2 libgbm1 libxcomposite1 libxdamage1 \
    libxrandr2 libappindicator3-1 libpangocairo-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Install Playwright browsers
RUN playwright install --with-deps

# Copy source
COPY . .

# Default command
CMD ["python", "crawler_agent.py"]
