# AI X-ray Diagnostic Tool

## Overview

This is a Flask-based web application that uses deep learning to analyze chest X-ray images for medical diagnosis. The system can classify X-rays into three categories (Normal, Pneumonia, COVID-19) and predict severity scores using a multi-task neural network approach.

## System Architecture

### Frontend Architecture
- **Template Engine**: Jinja2 templates with Bootstrap 5 for responsive UI
- **Styling**: Custom medical-themed CSS with dark theme support
- **JavaScript**: Client-side file upload handling, image preview, and form validation
- **UI Framework**: Bootstrap 5 with Font Awesome icons for a professional medical interface

### Backend Architecture
- **Web Framework**: Flask with SQLAlchemy for database operations
- **File Handling**: Werkzeug for secure file uploads with size and type validation
- **Production Server**: Gunicorn WSGI server for deployment
- **AI Model**: PyTorch-based multi-task neural network using ResNet18 backbone

### Database Schema
- **XrayPrediction Model**: Stores prediction results with metadata
  - Classification results (disease type: 0=Normal, 1=Pneumonia, 2=COVID-19)
  - Severity scoring (0.0 to 1.0 scale)
  - Confidence scoring
  - Image metadata (dimensions, file size)
  - Timestamps for tracking analysis history

## Key Components

### 1. Deep Learning Model (`model_utils.py`)
- **Architecture**: Multi-task ResNet18-based model
- **Tasks**: 
  - Classification head for disease detection
  - Regression head for severity prediction
- **Input Processing**: Torchvision transforms for image preprocessing
- **Output**: Disease classification + severity score (0-1)

### 2. Web Application (`app.py`, `routes.py`)
- **File Upload**: Secure handling of medical images (PNG, JPG, JPEG, TIFF, BMP)
- **Image Processing**: PIL-based image handling and validation
- **Results Display**: Comprehensive analysis results with confidence metrics
- **History Tracking**: Database storage of all previous predictions

### 3. Database Layer (`models.py`)
- **SQLAlchemy ORM**: Object-relational mapping for clean database interactions
- **Flexible Storage**: Supports both SQLite (development) and PostgreSQL (production)
- **Data Integrity**: Proper validation and constraints on medical data

### 4. User Interface
- **Responsive Design**: Mobile-first approach with Bootstrap grid system
- **Medical Theme**: Professional color scheme suitable for healthcare applications
- **Interactive Features**: Drag-and-drop upload, image preview, progress indicators

## Data Flow

1. **Image Upload**: User uploads chest X-ray through web interface
2. **File Validation**: System validates file type, size, and format
3. **Preprocessing**: Image is resized and normalized for model input
4. **AI Inference**: Multi-task model predicts disease class and severity
5. **Result Storage**: Predictions saved to database with metadata
6. **Display Results**: User sees classification, severity, and confidence scores
7. **History Access**: Previous analyses available through history page

## External Dependencies

### Core Dependencies
- **Flask Ecosystem**: Flask, Flask-SQLAlchemy for web framework
- **Deep Learning**: PyTorch, Torchvision for AI model
- **Image Processing**: Pillow for image handling
- **Scientific Computing**: NumPy, Scikit-learn for data operations
- **Visualization**: Matplotlib for potential charts and graphs

### Production Dependencies
- **WSGI Server**: Gunicorn for production deployment
- **Database**: psycopg2-binary for PostgreSQL support
- **Security**: Werkzeug for secure file handling

### Development Environment
- **Package Management**: UV for fast Python package resolution
- **PyTorch Index**: CPU-optimized PyTorch installation for Replit environment

## Deployment Strategy

### Development Environment
- **Platform**: Replit with Python 3.11
- **Database**: SQLite for local development
- **Server**: Flask development server with auto-reload

### Production Environment
- **WSGI Server**: Gunicorn with multi-worker configuration
- **Database**: PostgreSQL for production data storage
- **File Storage**: Local file system with organized upload directory
- **Scaling**: Autoscale deployment target for handling traffic spikes

### Configuration Management
- **Environment Variables**: 
  - `DATABASE_URL` for database connection
  - `SESSION_SECRET` for secure sessions
- **File Limits**: 16MB maximum upload size
- **Security**: ProxyFix middleware for proper header handling

## Changelog
- June 19, 2025: Enhanced AI model with feature-based analysis for more realistic disease identification
- June 19, 2025: Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.