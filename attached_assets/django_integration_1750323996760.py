# =====================================
# models.py - Django Models
# =====================================

from django.db import models
from django.contrib.auth.models import User

class XrayPrediction(models.Model):
    DISEASE_CHOICES = [
        (0, 'Normal'),
        (1, 'Pneumonia'),
        (2, 'COVID-19'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    image = models.ImageField(upload_to='xray_images/')
    predicted_disease = models.IntegerField(choices=DISEASE_CHOICES)
    severity_score = models.FloatField()
    confidence_score = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.get_predicted_disease_display()} - {self.severity_score:.2f}"

# =====================================
# forms.py - Django Forms
# =====================================

from django import forms
from .models import XrayPrediction

class XrayUploadForm(forms.ModelForm):
    class Meta:
        model = XrayPrediction
        fields = ['image']
        widgets = {
            'image': forms.ClearableFileInput(attrs={
                'class': 'form-control-file',
                'accept': 'image/*'
            })
        }

# =====================================
# model_utils.py - Model Prediction Logic
# =====================================

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import torch.nn.functional as F

class XrayMultiTaskModel(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super(XrayMultiTaskModel, self).__init__()
        self.backbone = models.resnet18(pretrained=pretrained)
        self.features = nn.Sequential(*list(self.backbone.children())[:-1])
        
        num_features = self.backbone.fc.in_features
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        self.regressor = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        
        classification_output = self.classifier(features)
        severity_output = self.regressor(features)
        
        return classification_output, severity_output

class XrayPredictor:
    def __init__(self, model_path='best_xray_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = XrayMultiTaskModel(num_classes=3)
        
        # Load trained weights
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
        except FileNotFoundError:
            print(f"Model file {model_path} not found. Please train the model first.")
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.class_names = ['Normal', 'Pneumonia', 'COVID-19']
    
    def predict(self, image_path):
        """
        Predict disease type and severity from X-ray image
        
        Returns:
            dict: {
                'predicted_class': int,
                'class_name': str,
                'confidence': float,
                'severity': float,
                'probabilities': list
            }
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                class_output, severity_output = self.model(image_tensor)
                
                # Get probabilities and predicted class
                probabilities = F.softmax(class_output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
                severity = severity_output.item()
                
                return {
                    'predicted_class': predicted_class,
                    'class_name': self.class_names[predicted_class],
                    'confidence': confidence,
                    'severity': severity,
                    'probabilities': probabilities[0].cpu().numpy().tolist()
                }
                
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None

# Global predictor instance
predictor = XrayPredictor()

# =====================================
# views.py - Django Views
# =====================================

from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
import json
import os
from .forms import XrayUploadForm
from .models import XrayPrediction
from .model_utils import predictor

def home(request):
    """Home page view"""
    return render(request, 'diagnosis/home.html')

def upload_xray(request):
    """Handle X-ray image upload and prediction"""
    if request.method == 'POST':
        form = XrayUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the uploaded image temporarily
            prediction_obj = form.save(commit=False)
            
            # Get the uploaded image path
            image_path = prediction_obj.image.path if hasattr(prediction_obj.image, 'path') else prediction_obj.image.url
            
            # Make prediction
            result = predictor.predict(image_path)
            
            if result:
                # Save prediction results
                prediction_obj.predicted_disease = result['predicted_class']
                prediction_obj.severity_score = result['severity']
                prediction_obj.confidence_score = result['confidence']
                
                if request.user.is_authenticated:
                    prediction_obj.user = request.user
                
                prediction_obj.save()
                
                # Prepare context for results page
                context = {
                    'prediction': prediction_obj,
                    'result': result,
                    'severity_percentage': int(result['severity'] * 100),
                    'confidence_percentage': int(result['confidence'] * 100),
                }
                
                return render(request, 'diagnosis/result.html', context)
            else:
                messages.error(request, 'Error processing the image. Please try again.')
                return redirect('upload_xray')
    else:
        form = XrayUploadForm()
    
    return render(request, 'diagnosis/upload.html', {'form': form})

@csrf_exempt
def api_predict(request):
    """API endpoint for predictions"""
    if request.method == 'POST':
        try:
            # Handle file upload via API
            if 'image' in request.FILES:
                image_file = request.FILES['image']
                
                # Save temporarily
                temp_path = f'/tmp/{image_file.name}'
                with open(temp_path, 'wb+') as destination:
                    for chunk in image_file.chunks():
                        destination.write(chunk)
                
                # Make prediction
                result = predictor.predict(temp_path)
                
                # Clean up temp file
                os.remove(temp_path)
                
                if result:
                    return JsonResponse({
                        'success': True,
                        'prediction': result
                    })
                else:
                    return JsonResponse({
                        'success': False,
                        'error': 'Failed to process image'
                    })
            else:
                return JsonResponse({
                    'success': False,
                    'error': 'No image provided'
                })
                
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    return JsonResponse({
        'success': False,
        'error': 'Only POST method allowed'
    })

def prediction_history(request):
    """Show user's prediction history"""
    if request.user.is_authenticated:
        predictions = XrayPrediction.objects.filter(user=request.user).order_by('-created_at')
    else:
        predictions = []
    
    return render(request, 'diagnosis/history.html', {'predictions': predictions})

# =====================================
# urls.py - URL Configuration
# =====================================

from django.urls import path
from . import views

app_name = 'diagnosis'

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_xray, name='upload_xray'),
    path('history/', views.prediction_history, name='history'),
    path('api/predict/', views.api_predict, name='api_predict'),
]

# =====================================
# settings.py additions
# =====================================

"""
Add these to your Django settings.py:

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'diagnosis',  # Your app
]

MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# For file uploads
FILE_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024  # 10MB
DATA_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024  # 10MB
"""

# =====================================
# requirements.txt
# =====================================

"""
Django>=4.0
torch>=1.9.0
torchvision>=0.10.0
Pillow>=8.0.0
numpy>=1.21.0
kagglehub>=0.1.0
tqdm>=4.62.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
"""