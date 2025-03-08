import os
import cv2
import uuid
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory, session
from werkzeug.utils import secure_filename # Secure filename to prevent path traversal attacks
from ultralytics import YOLO
import time
import logging
from datetime import datetime
import imghdr # validate image file types
import torch
from torch import serialization

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = os.path.join('static', 'uploads')
RESULT_FOLDER = os.path.join('static', 'images')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load YOLOv8 model
MODEL_PATH = os.path.join('models', 'yolov8n.pt')  # Default model path

# Define damage classes and severity mappings - would be replaced with actual classes from trained model
DAMAGE_CLASSES = {
    0: "scratch",
    1: "dent",
    2: "broken glass",
    3: "broken light",
    4: "major damage"
}

DAMAGE_SEVERITY = {
    "scratch": 1,
    "dent": 2,
    "broken glass": 3,
    "broken light": 3,
    "major damage": 4
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image(file_path):
    """
    Validate if the file is a proper image that can be processed
    Returns: (is_valid, message)
    """
    # Check if file exists
    if not os.path.exists(file_path):
        return False, "Image file not found"
    
    # Check file size
    if os.path.getsize(file_path) == 0:
        return False, "Image file is empty"
    
    # Check if it's a valid image format using imghdr
    img_type = imghdr.what(file_path)
    if img_type not in ['jpeg', 'png', 'jpg']:
        return False, f"Invalid image format: {img_type}"
    
    # Try to open with OpenCV to verify it's a valid image
    img = cv2.imread(file_path)
    if img is None:
        return False, "Cannot read image file - file may be corrupted"
    
    # Additional check for image dimensions
    height, width = img.shape[:2]
    if height < 10 or width < 10:
        return False, f"Image is too small ({width}x{height})"
    if height > 10000 or width > 10000:
        return False, f"Image is too large ({width}x{height})"
    
    return True, "Image is valid"

def load_model():
    try:
        # Configure PyTorch to allow YOLO model loading
        serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])
        
        if os.path.exists(MODEL_PATH):
            logger.info(f"Loading model from {MODEL_PATH}")
            try:
                # First try with weights_only=True (new default in PyTorch 2.6)
                model = YOLO(MODEL_PATH)
            except Exception as e:
                logger.warning(f"Failed to load with weights_only=True, trying with weights_only=False: {e}")
                # If that fails, try with weights_only=False since we trust our own model file
                model = torch.load(MODEL_PATH, weights_only=False)
            return model
        else:
            logger.info("Model not found, downloading YOLOv8 model")
            # If model doesn't exist, download a small YOLOv8 model
            model = YOLO('yolov8n.pt')
            os.makedirs('models', exist_ok=True)
            # Save with explicit torch.save to avoid pickling issues
            torch.save(model.model, MODEL_PATH)
            logger.info(f"Model saved to {MODEL_PATH}")
            return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

def clean_old_files(folder_path, max_age_days=1):
    """Clean up files older than max_age_days"""
    try:
        current_time = datetime.now()
        count = 0
        
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            if os.path.isfile(file_path):
                file_age = datetime.fromtimestamp(os.path.getctime(file_path))
                age_days = (current_time - file_age).days
                
                if age_days >= max_age_days:
                    os.remove(file_path)
                    count += 1
        
        if count > 0:
            logger.info(f"Cleaned {count} old files from {folder_path}")
    except Exception as e:
        logger.error(f"Error cleaning old files: {e}")

def delete_user_files(file_prefix):
    """Delete specific user files based on filename prefix"""
    try:
        deleted_count = 0
        
        # Delete from uploads folder
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            if filename.startswith(file_prefix):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    deleted_count += 1
        
        # Delete from results folder
        for filename in os.listdir(app.config['RESULT_FOLDER']):
            if filename.startswith(f"result_{file_prefix}") or filename.startswith(f"error_{file_prefix}"):
                file_path = os.path.join(app.config['RESULT_FOLDER'], filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    deleted_count += 1
        
        logger.info(f"Deleted {deleted_count} files with prefix {file_prefix}")
        return deleted_count
    except Exception as e:
        logger.error(f"Error deleting user files: {e}")
        return 0

def calculate_damage_severity(damage_types):
    """Calculate overall damage severity based on damage types"""
    if not damage_types:
        return "No damage detected"
    
    max_severity = 0
    total_severity = 0
    
    for damage_type in damage_types:
        severity = DAMAGE_SEVERITY.get(damage_type.lower(), 1)
        max_severity = max(max_severity, severity)
        total_severity += severity
    
    avg_severity = total_severity / len(damage_types)
    
    # Consider both maximum severity and average severity
    if max_severity >= 4:
        return "Severe damage"
    elif max_severity >= 3 or avg_severity > 2:
        return "Moderate damage"
    else:
        return "Minor damage"

@app.route('/')
def index():
    # Generate a unique session ID if not already present
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    
    # Clean up old files periodically
    try:
        clean_old_files(app.config['UPLOAD_FOLDER'], 1)
        clean_old_files(app.config['RESULT_FOLDER'], 1)
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    
    # Ensure user_id is in session
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    
    if file and allowed_file(file.filename):
        try:
            # Generate unique filename with session ID prefix
            user_id = session['user_id']
            filename = f"{user_id}_{secure_filename(file.filename)}"
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save the uploaded image
            file.save(upload_path)
            logger.info(f"File uploaded: {filename}")
            
            # Store filename in session for later cleanup
            if 'user_files' not in session:
                session['user_files'] = []
            session['user_files'].append(filename)
            
            # Validate image before processing
            is_valid, message = validate_image(upload_path)
            if not is_valid:
                logger.error(f"Image validation failed: {message}")
                flash(f"Invalid image: {message}")
                # Remove the invalid file
                try:
                    os.remove(upload_path)
                    logger.info(f"Removed invalid file: {filename}")
                except Exception as e:
                    logger.error(f"Error removing invalid file: {e}")
                return redirect(url_for('index'))
            
            # Process the image for damage detection
            result_path, analysis = detect_damage(upload_path, filename)
            
            return render_template('result.html', 
                                  original_image=upload_path, 
                                  result_image=result_path,
                                  analysis=analysis)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error processing upload: {error_msg}")
            flash(f"Error processing image: {error_msg}")
            return redirect(url_for('index'))
    else:
        flash('Allowed file types are png, jpg, jpeg')
        return redirect(url_for('index'))

@app.route('/cleanup', methods=['POST'])
def cleanup_files():
    """API endpoint to remove user files on page unload"""
    try:
        user_id = request.form.get('user_id') or session.get('user_id')
        if user_id:
            count = delete_user_files(user_id)
            logger.info(f"Cleanup requested for user {user_id}, deleted {count} files")
            return jsonify({"success": True, "deleted": count})
        return jsonify({"success": False, "error": "No user ID provided"})
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/download/<path:filename>')
def download_file(filename):
    """Allow downloading result images"""
    directory = os.path.dirname(filename)
    file = os.path.basename(filename)
    return send_from_directory(directory, file, as_attachment=True)

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large errors"""
    flash('File is too large. Maximum size is 16MB')
    return redirect(url_for('index')), 413

@app.errorhandler(500)
def internal_server_error(error):
    """Handle internal server errors"""
    logger.error(f"Server error: {error}")
    flash('An internal server error occurred. Please try again later.')
    return redirect(url_for('index')), 500

@app.errorhandler(404)
def page_not_found(error):
    """Handle page not found errors"""
    return render_template('404.html'), 404

def detect_damage(image_path, filename):
    try:
        logger.info(f"Starting damage detection for {filename}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Image path exists: {os.path.exists(image_path)}")
        
        # Validate the image a second time as a safeguard
        is_valid, message = validate_image(image_path)
        if not is_valid:
            logger.error(f"Image validation failed: {message}")
            raise Exception(f"Invalid image: {message}")
        
        # Load the model
        model = load_model()
        if not model:
            logger.error("Model loading failed")
            raise Exception("Failed to load model")
        
        # Create result image with bounding boxes
        logger.info("Reading image with OpenCV")
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"OpenCV failed to read image at {image_path}")
            raise Exception("Could not read the image")
            
        logger.info(f"Image shape: {img.shape}")
        result_img = img.copy()
        
        # Analysis dictionary
        analysis = {
            'damages_detected': 0,
            'damage_types': [],
            'damage_confidences': [],
            'severity': 'Unknown',
            'processing_time': "0.00",
            'image_dimensions': f"{img.shape[1]}x{img.shape[0]}"
        }
        
        try:
            # Process the image
            start_time = time.time()
            logger.info("Running model inference")
            results = model(image_path)
            processing_time = time.time() - start_time
            
            analysis['processing_time'] = f"{processing_time:.2f}"
            logger.info(f"Image processed in {processing_time:.2f} seconds")
            
            # Extract detection results
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Get confidence and class
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    # Only keep detections with confidence > 0.5
                    if conf > 0.5:
                        analysis['damages_detected'] += 1
                        
                        # Map class ID to damage type
                        damage_type = DAMAGE_CLASSES.get(cls % len(DAMAGE_CLASSES), "unknown damage")
                        
                        analysis['damage_types'].append(damage_type)
                        analysis['damage_confidences'].append(f"{conf:.2f}")
                        
                        # Draw bounding box
                        color = (0, 255, 0)  # Default green
                        
                        # Change color based on damage type
                        if "scratch" in damage_type:
                            color = (255, 255, 0)  # Cyan
                        elif "dent" in damage_type:
                            color = (0, 165, 255)  # Orange
                        elif "glass" in damage_type:
                            color = (0, 0, 255)  # Red
                        elif "light" in damage_type:
                            color = (255, 0, 0)  # Blue
                        elif "major" in damage_type:
                            color = (0, 0, 255)  # Red
                            
                        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
                        
                        # Add label with prettier formatting
                        label = f"{damage_type}: {conf:.2f}"
                        
                        # Calculate text size and add background
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(result_img, (x1, y1-text_size[1]-10), (x1+text_size[0], y1), color, -1)
                        cv2.putText(result_img, label, (x1, y1-5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        except Exception as e:
            logger.error(f"Error during model inference: {e}")
            # Even if model inference fails, we still want to return an image with error message
            # Add error text to the image
            cv2.putText(result_img, "Model processing error", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            analysis['error'] = f"Model error: {str(e)}"
        
        # Determine severity based on number and types of damages
        if analysis['damages_detected'] == 0:
            analysis['severity'] = 'No damage detected'
            
            # Add a message on the result image indicating no damage detected
            height, width = result_img.shape[:2]
            cv2.putText(result_img, "No damage detected", (width//2-150, height//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            analysis['severity'] = calculate_damage_severity(analysis['damage_types'])
        
        # Save result image
        result_filename = f"result_{filename}"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        cv2.imwrite(result_path, result_img)
        
        logger.info(f"Damage detection completed. Severity: {analysis['severity']}")
        
        return result_path, analysis
        
    except Exception as e:
        logger.error(f"Error in damage detection: {e}")
        
        # Create a simple error image
        try:
            # Try to load the original image if possible
            error_img = cv2.imread(image_path)
            if error_img is None:
                # If the original image can't be loaded, create a blank image
                error_img = np.ones((400, 600, 3), dtype=np.uint8) * 255
                
            # Add error text
            cv2.putText(error_img, "Error processing image", (50, 50), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(error_img, str(e), (50, 100), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Save the error image
            result_filename = f"error_{filename}"
            result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
            cv2.imwrite(result_path, error_img)
            
            # Create basic error analysis
            error_analysis = {
                'damages_detected': 0,
                'damage_types': [],
                'damage_confidences': [],
                'severity': 'Error',
                'processing_error': str(e),
                'error': True
            }
            
            return result_path, error_analysis
            
        except Exception as inner_error:
            logger.error(f"Failed to create error image: {inner_error}")
            raise Exception(f"Failed to process image: {str(e)}")

if __name__ == '__main__':
    try:
        # Check if model is available or download it
        load_model()
        # Start the Flask application
        app.run(debug=True)
    except Exception as e:
        logger.error(f"Application failed to start: {e}")