# Update package lists
apt-get update

# Install Tesseract OCR and any language packs you need (add more as needed)
apt-get install -y tesseract-ocr

# (Optional) Install additional language packs, e.g. Hindi and French
# apt-get install -y tesseract-ocr-hin tesseract-ocr-fra

# Clean up to reduce image size
apt-get clean
rm -rf /var/lib/apt/lists/*