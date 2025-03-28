import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Configuration settings
class Config:
    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    NUM_CLASSES = 4
    LEARNING_RATE = 0.0001
    EPOCHS = 20
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_DIR = 'preprocessed_data'
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    VALID_DIR = os.path.join(DATA_DIR, 'val')
    TEST_DIR = os.path.join(DATA_DIR, 'test')
    MODEL_PATH = 'updated_dense_net.pth'
    
# Check if CUDA is available
print("Using device:", Config.DEVICE)
if torch.cuda.is_available():
    print("GPU detected:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU")

# Data Transformations
train_transforms = transforms.Compose([
    transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(0, shear=20, scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom Model
class CustomDenseNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomDenseNet, self).__init__()
        # Use the correct weight specification
        self.model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        num_features = self.model.classifier.in_features
        
        # Custom classifier
        self.model.classifier = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

# Function to train the model
def train_model(model, dataloaders, criterion, optimizer, num_epochs):
    best_accuracy = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for inputs, labels in dataloaders['train']:
            # Move to the device (GPU or CPU)
            inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_accuracy = 100 * correct / total
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in dataloaders['valid']:
                # Move to the device (GPU or CPU)
                inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * val_correct / val_total
        print(f'Epoch {epoch+1}/{num_epochs} | Train Acc: {train_accuracy:.2f}% | Val Acc: {val_accuracy:.2f}%')
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f'New best model saved at {Config.MODEL_PATH}')
    
    return model

# Function to test the model
def test_model(model, dataloader, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in dataloader:
            # Move to the device (GPU or CPU)
            inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate overall accuracy
    accuracy = 100 * correct / total
    print(f'\nTest Accuracy: {accuracy:.2f}%')
    
    # Generate classification report
    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    print('\nConfusion Matrix:')
    print(conf_matrix)
    
    return accuracy

# Main execution
def main():
    # Load datasets
    train_dataset = datasets.ImageFolder(Config.TRAIN_DIR, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(Config.VALID_DIR, transform=test_transforms)
    test_dataset = datasets.ImageFolder(Config.TEST_DIR, transform=test_transforms)
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True),
        'valid': DataLoader(valid_dataset, batch_size=Config.BATCH_SIZE, shuffle=False),
        'test': DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    }
    
    # Get class names
    class_names = list(train_dataset.class_to_idx.keys())
    print("Class Indices:", train_dataset.class_to_idx)
    
    # Initialize model
    model = CustomDenseNet(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
    print("Model moved to:", next(model.parameters()).device)  # Confirm model device
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.model.classifier.parameters(), lr=Config.LEARNING_RATE)
    
    # Train model
    model = train_model(model, dataloaders, criterion, optimizer, Config.EPOCHS)
    
    # Load the best model
    model.load_state_dict(torch.load(Config.MODEL_PATH))
    
    # Test the model
    test_model(model, dataloaders['test'], class_names)
    
    print(f"Training and testing complete. Model saved as {Config.MODEL_PATH}")

if __name__ == '__main__':
    main()
