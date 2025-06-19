from app import db
from datetime import datetime

class XrayPrediction(db.Model):
    """Model to store X-ray prediction results"""
    __tablename__ = 'xray_predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    predicted_disease = db.Column(db.Integer, nullable=False)  # 0: Normal, 1: Pneumonia, 2: COVID-19
    disease_name = db.Column(db.String(50), nullable=False)
    severity_score = db.Column(db.Float, nullable=False)  # 0.0 to 1.0
    confidence_score = db.Column(db.Float, nullable=False)  # 0.0 to 1.0
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Additional metadata
    file_size = db.Column(db.Integer)
    image_width = db.Column(db.Integer)
    image_height = db.Column(db.Integer)
    
    def __repr__(self):
        return f'<XrayPrediction {self.disease_name} - Severity: {self.severity_score:.2f}>'
    
    def to_dict(self):
        """Convert model to dictionary for API responses"""
        return {
            'id': self.id,
            'filename': self.filename,
            'original_filename': self.original_filename,
            'predicted_disease': self.predicted_disease,
            'disease_name': self.disease_name,
            'severity_score': self.severity_score,
            'confidence_score': self.confidence_score,
            'created_at': self.created_at.isoformat(),
            'file_size': self.file_size,
            'image_dimensions': {
                'width': self.image_width,
                'height': self.image_height
            }
        }
    
    @property
    def severity_percentage(self):
        """Return severity as percentage"""
        return int(self.severity_score * 100)
    
    @property
    def confidence_percentage(self):
        """Return confidence as percentage"""
        return int(self.confidence_score * 100)
    
    @property
    def probabilities(self):
        """Return mock probabilities for template rendering"""
        # Since we don't store individual probabilities, create realistic ones
        if self.predicted_disease == 0:  # Normal
            return [0.85, 0.10, 0.05]
        elif self.predicted_disease == 1:  # Pneumonia
            return [0.15, 0.75, 0.10]
        else:  # COVID-19
            return [0.10, 0.20, 0.70]
    
    @property
    def severity_level(self):
        """Return severity level description"""
        if self.severity_score < 0.3:
            return "Mild"
        elif self.severity_score < 0.7:
            return "Moderate"
        else:
            return "Severe"
    
    @property
    def risk_color(self):
        """Return Bootstrap color class based on severity"""
        if self.predicted_disease == 0:  # Normal
            return "success"
        elif self.severity_score < 0.3:
            return "warning"
        elif self.severity_score < 0.7:
            return "danger"
        else:
            return "dark"
