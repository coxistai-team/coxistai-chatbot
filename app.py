import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from modules.query import SmartDeepSeek
from modules.text_classifier import is_educational
from modules.pdf_parser import extract_text_from_file
from modules.image_ocr import extract_text_from_image

load_dotenv()

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'temp_uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_KEY = os.getenv("OPENAI_API_KEY")
assistant = SmartDeepSeek(API_KEY)

SYSTEM_PROMPT = """You are an expert educational assistant named SparkTutor. Provide clean, well-structured, and direct answers.
- Use bold text for key terms.
- Use bullet points or numbered lists where appropriate for clarity.
- Do not repeat the user's question in your response.
"""

ALLOWED_EXTENSIONS = {
    'image': {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'},
    'document': {'pdf', 'docx'},
    'audio': {'mp3', 'wav', 'm4a'}
}

def allowed_file(filename, file_type):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS.get(file_type, set())

def extract_text_from_file_input(file_path, file_type):
    try:
        if file_type == 'image':
            return extract_text_from_image(file_path), True
        elif file_type == 'document':
            return extract_text_from_file(file_path)
        else:
            return None, False
    except Exception as e:
        logger.error(f"Error extracting text from {file_type}: {str(e)}")
        return None, False

@app.route('/')
def index():
    return jsonify({'message': 'SparkTutor API is running'})

@app.route('/api/health', methods=['GET'])
def health_check():
    serializable_extensions = {
        key: list(value) for key, value in ALLOWED_EXTENSIONS.items()
    }
    return jsonify({
        'status': 'healthy',
        'supported_files': serializable_extensions
    })

@app.route('/api/chat/text', methods=['POST'])
def chat_text():
    data = request.get_json()
    if not data or 'message' not in data or not data['message'].strip():
        return jsonify({'error': 'Message is required and cannot be empty'}), 400

    message = data['message'].strip()
    if not is_educational(message):
        return jsonify({
            'success': True,
            'ai_response': "I specialize in educational content. Please ask about academic subjects.",
            'is_educational': False,
        })
    
    try:
        response = assistant.get_response(message, system_prompt=SYSTEM_PROMPT)
        return jsonify({'success': True, 'ai_response': response, 'is_educational': True})
    except Exception as e:
        logger.error(f"Error in chat_text: {str(e)}")
        return jsonify({'error': 'Failed to generate response.'}), 500

@app.route('/api/chat/file', methods=['POST'])
def chat_file():
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'error': 'No file provided or selected'}), 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    file_type = next((ftype for ftype, exts in ALLOWED_EXTENSIONS.items() if allowed_file(filename, ftype)), None)

    if not file_type:
        return jsonify({'error': 'Unsupported file type'}), 400

    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        file.save(temp_path)
        extracted_text, success = extract_text_from_file_input(temp_path, file_type)
        
        if not success or not extracted_text:
            return jsonify({'error': f'Failed to extract text from {file_type} file.'}), 400
        
        if not is_educational(extracted_text):
            return jsonify({
                'success': True,
                'ai_response': "This content doesn't appear to be educational.",
                'is_educational': False
            })

        response = assistant.get_response(extracted_text, system_prompt=SYSTEM_PROMPT)
        return jsonify({'success': True, 'ai_response': response, 'is_educational': True})

    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return jsonify({'error': 'Failed to process file'}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/api/classify', methods=['POST'])
def classify_text():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Text is required'}), 400
    
    text = data['text']
    return jsonify({
        'text': text[:200] + '...' if len(text) > 200 else text,
        'is_educational': is_educational(text)
    })

@app.route('/api/extract', methods=['POST'])
def extract_only():
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'error': 'No file provided or selected'}), 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    file_type = next((ftype for ftype, exts in ALLOWED_EXTENSIONS.items() if allowed_file(filename, ftype)), None)

    if not file_type:
        return jsonify({'error': 'Unsupported file type'}), 400
    
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        file.save(temp_path)
        extracted_text, success = extract_text_from_file_input(temp_path, file_type)
        if success and extracted_text:
            return jsonify({'success': True, 'extracted_text': extracted_text})
        else:
            return jsonify({'success': False, 'error': 'Failed to extract text'}), 400
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3001)