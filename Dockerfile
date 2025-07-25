FROM python:3.12-slim

# Install system dependencies (Tesseract OCR and poppler for pdf2image)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        poppler-utils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

# Expose the port your app runs on
EXPOSE 3001

# Start the app with gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:3001"]