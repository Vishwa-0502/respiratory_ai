import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import kagglehub
import os
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Custom Dataset Class
class XrayDataset(Dataset):
    def __init__(self, image_paths, labels, severities, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.severities = severities
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        severity = torch.tensor(self.severities[idx], dtype=torch.float32)
        
        return image, label, severity

# Multi-task Model Architecture
class XrayMultiTaskModel(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super(XrayMultiTaskModel, self).__init__()
        # Use ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Remove the final classification layer
        self.features = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Get the number of features from the backbone
        num_features = self.backbone.fc.in_features
        
        # Classification head (3 classes: Normal, Pneumonia, COVID-19)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        # Severity regression head (0-1 scale)
        self.regressor = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
    
    def forward(self, x):
        # Extract features
        features = self.features(x)
        features = features.view(features.size(0), -1)  # Flatten
        
        # Get predictions from both heads
        classification_output = self.classifier(features)
        severity_output = self.regressor(features)
        
        return classification_output, severity_output

# Data preprocessing and augmentation
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_test_transform

# Dataset preparation function
def prepare_datasets():
    """
    Download and prepare datasets from Kaggle
    """
    print("Downloading datasets...")
    
    # Download datasets using kagglehub
    dataset1_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    dataset2_path = kagglehub.dataset_download("prashant268/chest-xray-covid19-pneumonia")
    
    print(f"Dataset 1 path: {dataset1_path}")
    print(f"Dataset 2 path: {dataset2_path}")
    
    # You'll need to implement the dataset merging logic here
    # This is a placeholder for the actual implementation
    image_paths = []
    labels = []  # 0: Normal, 1: Pneumonia, 2: COVID-19
    severities = []  # 0.0-1.0 scale
    
    # Example logic (you'll need to adapt based on actual dataset structure):
    # - Scan through dataset directories
    # - Assign labels based on folder names
    # - Assign severity scores (you might need to create these based on metadata)
    
    return image_paths, labels, severities

# Training function
def train_model(model, train_loader, val_loader, num_epochs=30, device='cuda'):
    # Loss functions
    criterion_class = nn.CrossEntropyLoss()
    criterion_severity = nn.MSELoss()
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    # Training history
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for images, labels, severities in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images, labels, severities = images.to(device), labels.to(device), severities.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            class_outputs, severity_outputs = model(images)
            
            # Calculate losses
            class_loss = criterion_class(class_outputs, labels)
            severity_loss = criterion_severity(severity_outputs.squeeze(), severities)
            
            # Combined loss (you can adjust weights)
            total_loss = class_loss + 0.5 * severity_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels, severities in val_loader:
                images, labels, severities = images.to(device), labels.to(device), severities.to(device)
                
                class_outputs, severity_outputs = model(images)
                
                class_loss = criterion_class(class_outputs, labels)
                severity_loss = criterion_severity(severity_outputs.squeeze(), severities)
                total_loss = class_loss + 0.5 * severity_loss
                
                val_loss += total_loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(class_outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Calculate averages
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_xray_model.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break
    
    return train_losses, val_losses, val_accuracies

# Main training script
def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Prepare datasets
    print("Preparing datasets...")
    image_paths, labels, severities = prepare_datasets()
    
    # Split data
    train_paths, temp_paths, train_labels, temp_labels, train_severities, temp_severities = train_test_split(
        image_paths, labels, severities, test_size=0.3, random_state=42, stratify=labels
    )
    
    val_paths, test_paths, val_labels, test_labels, val_severities, test_severities = train_test_split(
        temp_paths, temp_labels, temp_severities, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    # Get transforms
    train_transform, val_test_transform = get_transforms()
    
    # Create datasets
    train_dataset = XrayDataset(train_paths, train_labels, train_severities, train_transform)
    val_dataset = XrayDataset(val_paths, val_labels, val_severities, val_test_transform)
    test_dataset = XrayDataset(test_paths, test_labels, test_severities, val_test_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Initialize model
    model = XrayMultiTaskModel(num_classes=3).to(device)
    
    # Train model
    print("Starting training...")
    train_losses, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, num_epochs=30, device=device
    )
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.title('Validation Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    print("Training completed! Best model saved as 'best_xray_model.pth'")

if __name__ == "__main__":
    main()
