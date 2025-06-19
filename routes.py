from flask import render_template, request, jsonify, flash, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
import uuid
from datetime import datetime
import logging

from app import app, db
from models import XrayPrediction
from model_utils import predictor

logger = logging.getLogger(__name__)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'bmp'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/upload')
def upload_page():
    """Upload page"""
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction"""
    try:
        # Check if file was uploaded
        if 'xray_image' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['xray_image']
        
        # Check if file was actually selected
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        # Validate file type
        if not allowed_file(file.filename):
            flash('Invalid file type. Please upload PNG, JPG, JPEG, TIFF, or BMP files.', 'error')
            return redirect(request.url)
        
        if file:
            # Generate unique filename
            original_filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{original_filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            # Save uploaded file
            file.save(file_path)
            logger.info(f"File saved: {file_path}")
            
            # Get image information
            image_info = predictor.get_image_info(file_path)
            
            # Make prediction
            result = predictor.predict(file_path)
            
            if result['success']:
                # Save prediction to database
                prediction = XrayPrediction(
                    filename=unique_filename,
                    original_filename=original_filename,
                    predicted_disease=result['predicted_class'],
                    disease_name=result['class_name'],
                    severity_score=result['severity'],
                    confidence_score=result['confidence'],
                    file_size=image_info['file_size'] if image_info else None,
                    image_width=image_info['width'] if image_info else None,
                    image_height=image_info['height'] if image_info else None
                )
                
                db.session.add(prediction)
                db.session.commit()
                
                logger.info(f"Prediction saved: {prediction}")
                
                return redirect(url_for('results', prediction_id=prediction.id))
            else:
                # Clean up uploaded file if prediction failed
                if os.path.exists(file_path):
                    os.remove(file_path)
                
                flash(f"Prediction failed: {result['error']}", 'error')
                return redirect(request.url)
                
    except Exception as e:
        logger.error(f"Error in upload: {e}")
        flash(f"An error occurred: {str(e)}", 'error')
        return redirect(request.url)

@app.route('/results/<int:prediction_id>')
def results(prediction_id):
    """Display prediction results"""
    prediction = XrayPrediction.query.get_or_404(prediction_id)
    return render_template('results.html', prediction=prediction)

@app.route('/history')
def history():
    """Display prediction history"""
    page = request.args.get('page', 1, type=int)
    predictions = XrayPrediction.query.order_by(XrayPrediction.created_at.desc()).paginate(
        page=page, per_page=10, error_out=False
    )
    return render_template('history.html', predictions=predictions)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# API Endpoints
@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        # Check if file was uploaded
        if 'xray_image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        file = request.files['xray_image']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file type'
            }), 400
        
        # Save file temporarily
        unique_filename = f"temp_{uuid.uuid4()}_{secure_filename(file.filename)}"
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(temp_path)
        
        try:
            # Make prediction
            result = predictor.predict(temp_path)
            
            if result['success']:
                # Get image info
                image_info = predictor.get_image_info(temp_path)
                
                # Save to database (optional for API)
                save_to_db = request.form.get('save_to_db', 'false').lower() == 'true'
                
                prediction_data = {
                    'predicted_class': result['predicted_class'],
                    'disease_name': result['class_name'],
                    'severity_score': result['severity'],
                    'confidence_score': result['confidence'],
                    'severity_percentage': int(result['severity'] * 100),
                    'confidence_percentage': int(result['confidence'] * 100),
                    'probabilities': {
                        'Normal': result['probabilities'][0],
                        'Pneumonia': result['probabilities'][1],
                        'COVID-19': result['probabilities'][2]
                    }
                }
                
                if save_to_db:
                    prediction = XrayPrediction(
                        filename=unique_filename,
                        original_filename=secure_filename(file.filename),
                        predicted_disease=result['predicted_class'],
                        disease_name=result['class_name'],
                        severity_score=result['severity'],
                        confidence_score=result['confidence'],
                        file_size=image_info['file_size'] if image_info else None,
                        image_width=image_info['width'] if image_info else None,
                        image_height=image_info['height'] if image_info else None
                    )
                    
                    db.session.add(prediction)
                    db.session.commit()
                    
                    prediction_data['id'] = prediction.id
                else:
                    # Clean up temp file if not saving to DB
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                
                return jsonify({
                    'success': True,
                    'prediction': prediction_data
                })
            else:
                return jsonify(result), 400
                
        finally:
            # Clean up temp file if it still exists
            if not request.form.get('save_to_db', 'false').lower() == 'true' and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
                    
    except Exception as e:
        logger.error(f"Error in API predict: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/history')
def api_history():
    """API endpoint to get prediction history"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 10, type=int), 100)  # Max 100 per page
        
        predictions = XrayPrediction.query.order_by(XrayPrediction.created_at.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        return jsonify({
            'success': True,
            'predictions': [p.to_dict() for p in predictions.items],
            'pagination': {
                'page': predictions.page,
                'pages': predictions.pages,
                'per_page': predictions.per_page,
                'total': predictions.total,
                'has_next': predictions.has_next,
                'has_prev': predictions.has_prev
            }
        })
        
    except Exception as e:
        logger.error(f"Error in API history: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/prediction/<int:prediction_id>')
def api_get_prediction(prediction_id):
    """API endpoint to get specific prediction"""
    try:
        prediction = XrayPrediction.query.get_or_404(prediction_id)
        return jsonify({
            'success': True,
            'prediction': prediction.to_dict()
        })
    except Exception as e:
        logger.error(f"Error getting prediction: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Error handlers
@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    flash('File too large. Please upload files smaller than 16MB.', 'error')
    return redirect(url_for('upload_page'))

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    db.session.rollback()
    return render_template('500.html'), 500
