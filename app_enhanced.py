"""
GroomAI - Enhanced Flask Application
Advanced AI-Powered Skin Analysis & Care Recommendations
With robust model loading, error handling, and production optimizations
"""

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
import cv2
import mediapipe as mp
import json
import logging
import sys
from datetime import datetime
import traceback
from typing import Optional, Dict, Any

# Import our custom model loader
from model_loader import load_groomai_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('groomai.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app with enhanced configuration
app = Flask(__name__)

# Configuration
class Config:
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'gif'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    MODEL_PATH = r'C:\Users\Administrator\Downloads\GroomAI-model\model\groomai_skin_model.h5'
    SECRET_KEY = os.environ.get('SECRET_KEY', 'groomai-secret-key-2025')
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

app.config.from_object(Config)

# Global model variable
model = None
model_info = {}

# Enhanced skin type information - adapted for 4-class model
skin_type_info = {
    0: {
        "type": "Type I-II — Fair/Light",
        "description": "Fair to light skin with pink to neutral undertones, burns easily, tans minimally",
        "care": "🧴 Essential Care:\n• Use SPF 50+ daily, reapply every 2 hours\n• Avoid sun exposure between 10am-4pm\n• Use gentle, fragrance-free cleansers\n• Apply ceramide-rich moisturizers\n• Include antioxidant serums (Vitamin C, E)\n• Weekly hydrating masks recommended\n• Avoid harsh exfoliants\n• Focus on barrier repair ingredients"
    },
    1: {
        "type": "Type III — Medium/Beige",
        "description": "Medium skin with warm undertones, moderate sun sensitivity, tans gradually",
        "care": "🧴 Essential Care:\n• SPF 30-50 daily application\n• Hyaluronic acid serums for hydration\n• Weekly gentle exfoliation (AHA/BHA)\n• Oil-balancing moisturizers\n• Retinol for anti-aging (PM only)\n• Clay masks for pore refinement\n• Vitamin C for brightness\n• Niacinamide for texture improvement"
    },
    2: {
        "type": "Type IV-V — Olive to Brown",
        "description": "Olive to brown skin with warm undertones, tans easily, rarely burns",
        "care": "🧴 Essential Care:\n• SPF 30+ daily protection\n• Niacinamide for pigmentation control\n• Lactic acid for gentle exfoliation\n• Lightweight gel moisturizers\n• Vitamin C for even tone\n• Weekly enzyme masks\n• Azelaic acid for dark spots\n• Glycolic acid for skin renewal"
    },
    3: {
        "type": "Type V-VI — Deep/Very Deep",
        "description": "Deep brown to black skin, highly melanin-rich, excellent natural sun protection",
        "care": "🧴 Essential Care:\n• SPF 30+ for UV protection\n• Mandelic acid for gentle exfoliation\n• Heavy moisturizers with shea butter\n• Kojic acid for hyperpigmentation\n• Vitamin C (stable forms like magnesium ascorbyl phosphate)\n• Glycerin-based hydrators\n• Weekly moisturizing masks\n• Arbutin for dark spot correction"
    }
}

def load_model():
    """Load the GroomAI model using our enhanced loader"""
    global model, model_info
    
    logger.info("Loading GroomAI model...")
    try:
        model = load_groomai_model(Config.MODEL_PATH)
        if model:
            logger.info("✅ Model loaded successfully!")
            # Get model information
            try:
                model_info = {
                    "input_shape": str(model.input_shape),
                    "output_shape": str(model.output_shape),
                    "total_params": model.count_params(),
                    "layers": len(model.layers)
                }
                logger.info(f"Model info: {model_info}")
            except:
                model_info = {"status": "info_unavailable"}
        else:
            logger.error("❌ Failed to load model")
            model_info = {"status": "failed"}
    except Exception as e:
        logger.error(f"❌ Exception during model loading: {e}")
        logger.error(traceback.format_exc())
        model = None
        model_info = {"status": "error", "error": str(e)}

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def detect_face(filepath: str) -> bool:
    """Detect face in image using MediaPipe"""
    try:
        mp_face_detection = mp.solutions.face_detection
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detector:
            img = cv2.imread(filepath)
            if img is None:
                logger.warning(f"Could not load image: {filepath}")
                return False
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_detector.process(img_rgb)
            
            face_detected = results.detections is not None and len(results.detections) > 0
            logger.info(f"Face detection result: {face_detected}")
            return face_detected
    except Exception as e:
        logger.error(f"Face detection error: {e}")
        return False

def preprocess_image(filepath: str) -> Optional[np.ndarray]:
    """Preprocess image for model prediction"""
    try:
        from tensorflow.keras.preprocessing import image
        
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        logger.info(f"Image preprocessed successfully. Shape: {img_array.shape}")
        return img_array
    except Exception as e:
        logger.error(f"Image preprocessing error: {e}")
        return None

def predict_skin_type(img_array: np.ndarray) -> Optional[Dict[str, Any]]:
    """Predict skin type using the loaded model"""
    global model
    
    if model is None:
        logger.error("Model not available for prediction")
        return None
    
    try:
        predictions = model.predict(img_array, verbose=0)
        predicted_index = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_index])
        
        logger.info(f"Prediction successful: Type {predicted_index}, Confidence: {confidence:.3f}")
        
        return {
            "predicted_index": predicted_index,
            "confidence": confidence,
            "predictions": predictions[0].tolist()
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return None

@app.route('/test')
def test_upload():
    """Simple test upload page for debugging"""
    from flask import send_file
    return send_file('test_upload.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    global model, model_info
    
    status = {
        "status": "healthy" if model is not None else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "model_info": model_info
    }
    
    return jsonify(status), 200 if model is not None else 503

@app.route('/', methods=['GET', 'POST'])
def index():
    """Main route for skin analysis"""
    if request.method == 'GET':
        return render_template('index.html')
    
    # POST request - handle image upload and analysis
    prediction_result = None
    error_message = None
    
    try:
        # Check if model is loaded
        if model is None:
            error_message = "⚠️ Model not available. Please try again later."
            logger.error("Model not available for analysis")
            return render_template('index.html', prediction=error_message)
        
        # Validate file upload
        if 'file' not in request.files:
            error_message = "⚠️ No file uploaded. Please select an image."
            logger.warning("No file uploaded")
            return render_template('index.html', prediction=error_message)
        
        file = request.files['file']
        if file.filename == '' or file.filename is None:
            error_message = "⚠️ No file selected. Please choose an image."
            return render_template('index.html', prediction=error_message)
        
        if not allowed_file(file.filename):
            error_message = "⚠️ Invalid file format. Please upload a JPG, PNG, WEBP, or GIF image."
            return render_template('index.html', prediction=error_message)
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"Processing uploaded file: {filename}")
        
        try:
            # Step 1: Face detection
            if not detect_face(filepath):
                error_message = "⚠️ No human face detected. Please upload a clear photo of your face."
                return render_template('index.html', prediction=error_message)
            
            # Step 2: Image preprocessing
            img_array = preprocess_image(filepath)
            if img_array is None:
                error_message = "⚠️ Error processing image. Please try with a different image."
                return render_template('index.html', prediction=error_message)
            
            # Step 3: Skin type prediction
            prediction_data = predict_skin_type(img_array)
            if prediction_data is None:
                error_message = "⚠️ Error during skin analysis. Please try again."
                return render_template('index.html', prediction=error_message)
            
            # Step 4: Format results
            predicted_index = prediction_data["predicted_index"]
            confidence = prediction_data["confidence"]
            confidence_percentage = round(confidence * 100, 1)
            
            if confidence < 0.6:
                error_message = f"⚠️ Analysis confidence too low ({confidence_percentage}%). Please upload a clearer, well-lit photo of your face."
                return render_template('index.html', prediction=error_message)
            
            # Get skin type information
            skin_info = skin_type_info.get(predicted_index, {
                "type": "Unknown Type",
                "description": "Unable to determine skin type.",
                "care": "Please consult with a dermatologist for personalized advice."
            })
            
            # Format final prediction
            prediction_result = f"""{skin_info['type']}

{skin_info['description']}

{skin_info['care']}

Confidence: {confidence_percentage}%
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
            
            logger.info(f"Analysis completed successfully: {skin_info['type']} ({confidence_percentage}%)")
            
        finally:
            # Clean up uploaded file
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    logger.info(f"Cleaned up uploaded file: {filename}")
            except Exception as e:
                logger.warning(f"Failed to clean up file {filename}: {e}")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        logger.error(traceback.format_exc())
        error_message = "⚠️ An unexpected error occurred. Please try again."
    
    # Return result
    result = prediction_result if prediction_result else error_message
    return render_template('index.html', prediction=result)

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for skin analysis"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                "success": False,
                "error": "Model not available",
                "error_code": "MODEL_UNAVAILABLE"
            }), 503
        
        # Validate file upload
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "error": "No file uploaded",
                "error_code": "NO_FILE"
            }), 400
        
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "error": "Invalid file format",
                "error_code": "INVALID_FORMAT"
            }), 400
        
        # Process file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Face detection
            if not detect_face(filepath):
                return jsonify({
                    "success": False,
                    "error": "No face detected",
                    "error_code": "NO_FACE"
                }), 400
            
            # Image preprocessing and prediction
            img_array = preprocess_image(filepath)
            if img_array is None:
                return jsonify({
                    "success": False,
                    "error": "Image processing failed",
                    "error_code": "PROCESSING_ERROR"
                }), 400
            
            prediction_data = predict_skin_type(img_array)
            if prediction_data is None:
                return jsonify({
                    "success": False,
                    "error": "Prediction failed",
                    "error_code": "PREDICTION_ERROR"
                }), 500
            
            # Format response
            predicted_index = prediction_data["predicted_index"]
            confidence = prediction_data["confidence"]
            
            if confidence < 0.6:
                return jsonify({
                    "success": False,
                    "error": "Low confidence",
                    "error_code": "LOW_CONFIDENCE",
                    "confidence": round(confidence * 100, 1)
                }), 400
            
            skin_info = skin_type_info.get(predicted_index, {})
            
            return jsonify({
                "success": True,
                "skin_type": skin_info.get('type', 'Unknown'),
                "description": skin_info.get('description', ''),
                "care_tips": skin_info.get('care', ''),
                "confidence": round(confidence * 100, 1),
                "type_index": predicted_index,
                "timestamp": datetime.now().isoformat(),
                "predictions": prediction_data["predictions"]
            }), 200
            
        finally:
            # Clean up
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except:
                pass
                
    except Exception as e:
        logger.error(f"API error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": "Internal server error",
            "error_code": "INTERNAL_ERROR"
        }), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({
        "success": False,
        "error": "File too large. Maximum size is 16MB.",
        "error_code": "FILE_TOO_LARGE"
    }), 413

@app.errorhandler(404)
def not_found(e):
    """Handle not found error"""
    return jsonify({
        "success": False,
        "error": "Endpoint not found",
        "error_code": "NOT_FOUND"
    }), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server error"""
    logger.error(f"Internal server error: {e}")
    return jsonify({
        "success": False,
        "error": "Internal server error",
        "error_code": "INTERNAL_ERROR"
    }), 500

# Initialize application
def create_app():
    """Create and configure the Flask application"""
    
    # Create upload directory
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    logger.info(f"Upload directory: {app.config['UPLOAD_FOLDER']}")
    
    # Load the model
    load_model()
    
    logger.info("GroomAI Flask application initialized successfully!")
    return app

if __name__ == '__main__':
    # Create the app
    app = create_app()
    
    # Run the application
    logger.info("🚀 Starting GroomAI server...")
    logger.info(f"Debug mode: {app.config['DEBUG']}")
    logger.info(f"Model status: {'Loaded' if model else 'Not loaded'}")
    
    try:
        app.run(
            debug=app.config['DEBUG'],
            host='0.0.0.0',
            port=int(os.environ.get('PORT', 5000)),
            threaded=True
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)
