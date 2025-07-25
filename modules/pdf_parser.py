import os
import sys
import logging
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from docx import Document

logging.basicConfig(level=logging.INFO)

class PDFInputHandler:
    def extract_text(self, pdf_path):
        try:
            with fitz.open(pdf_path) as doc:
                text = "\n".join(page.get_text() for page in doc)
            if len(text.strip()) >= 50:
                logging.info("Extracted text using PyMuPDF.")
                return text.strip(), True
        except Exception as e:
            logging.warning(f"Text-based extraction failed: {e}")

        try:
            logging.info("Trying OCR...")
            pages = convert_from_path(pdf_path)
            text = ""
            for i, page in enumerate(pages):
                ocr_text = pytesseract.image_to_string(page)
                text += f"\n--- Page {i + 1} ---\n{ocr_text}"
            return text.strip(), True if text.strip() else False
        except Exception as e:
            logging.error(f"OCR failed: {e}")
            return None, False


class DocxInputHandler:
    def extract_text(self, docx_path):
        try:
            doc = Document(docx_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text.strip(), True if text.strip() else False
        except Exception as e:
            logging.error(f"DOCX extraction failed: {e}")
            return None, False


def extract_text_from_file(file_path):
    if file_path.endswith(".pdf"):
        handler = PDFInputHandler()
    elif file_path.endswith(".docx"):
        handler = DocxInputHandler()
    else:
        logging.error("Unsupported file format.")
        return None, False

    return handler.extract_text(file_path)


def main():
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <file_path>")
        return

    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    text, success = extract_text_from_file(file_path)

    if success:
        print(f"\nExtracted {len(text)} characters.")
        print("First 500 chars preview:\n")
        print(text[:500])
    else:
        print("Extraction failed.")


if __name__ == "__main__":
    main()

        

