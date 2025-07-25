import pytesseract
from PIL import Image
import logging



logging.info("Pytesseract OCR module initialized.")

def extract_text_from_image(file_data) -> str:
    """
    Extracts text from an image using pytesseract.
    This function can accept either a file path (str) or file bytes.
    """
    try:
        # Use PIL (Pillow) to open the image from a path or from bytes
        image = Image.open(file_data)
        
        # Use pytesseract to perform OCR on the image
        text = pytesseract.image_to_string(image)
        
        return text.strip()

    except Exception as e:
        logging.error(f"Error during Pytesseract OCR extraction: {e}")
        return ""