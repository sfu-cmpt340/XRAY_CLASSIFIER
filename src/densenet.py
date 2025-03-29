import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import copy

# Configurations
class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TRAIN_DIR = "preprocessed_data/train"
    VALID_DIR = "preprocessed_data/val"
    TEST_DIR = "preprocessed_data/test"
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001
    EPOCHS = 15
    NUM_CLASSES = 4
    MODEL_PATH = "new_best_densenet_model.pth"
    MODEL_FULL_PATH = "new_best_densenet_full_model.pth"
    PATIENCE = 5  # For early stopping
    WEIGHT_DECAY = 1e-4  # L2 regularization

# Enhanced transformations - more augmentation to prevent overfitting
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),  # Add rotation
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Add translation
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Add color jitter
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder(Config.TRAIN_DIR, transform=train_transforms)
valid_dataset = datasets.ImageFolder(Config.VALID_DIR, transform=test_transforms)
test_dataset = datasets.ImageFolder(Config.TEST_DIR, transform=test_transforms)

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True),
    'valid': DataLoader(valid_dataset, batch_size=Config.BATCH_SIZE, shuffle=False),
    'test': DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
}

# Compute dynamic class weights
def get_class_weights(dataset):
    labels = [label for _, label in dataset.samples]
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    return torch.tensor(class_weights, dtype=torch.float).to(Config.DEVICE)

train_weights = get_class_weights(train_dataset)
criterion = nn.CrossEntropyLoss(weight=train_weights)

# Define DenseNet Model with Regularization
from torchvision import models

def create_model():
    # Use weights instead of pretrained parameter
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    
    # Freeze early layers to prevent overfitting
    for param in list(model.parameters())[:-100]:  # Freeze all but the last few layers
        param.requires_grad = False
    
    # Add dropout for regularization
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),  # Add dropout
        nn.Linear(num_ftrs, Config.NUM_CLASSES)
    )
    return model.to(Config.DEVICE)

model = create_model()

# Use optimizer with weight decay (L2 regularization)
optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)

# Learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

# Training function with early stopping
def train_model(model, dataloaders, criterion, optimizer, scheduler, epochs, patience):
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_model = None
    best_epoch = 0
    
    # For early stopping
    no_improve_epochs = 0
    
    start_time = time.time()
    
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(Config.DEVICE)
                labels = labels.to(Config.DEVICE)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Stats
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # If we are in validation phase, update scheduler
            if phase == 'valid':
                scheduler.step(epoch_loss)
                
                # Implement early stopping and save best model
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_model = copy.deepcopy(model)  # Save the entire model
                    best_epoch = epoch
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
        
        # Check for early stopping
        if no_improve_epochs >= patience:
            print(f"Early stopping triggered after epoch {epoch+1}. No improvement for {patience} epochs.")
            break
            
        print()
    
    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f} at epoch {best_epoch+1}')
    
    # Load best model
    model.load_state_dict(best_model_wts)
    
    torch.save({
        'epoch': best_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc
    }, Config.MODEL_PATH)
    
    # Save full model (easier for frontend deployment)
    torch.save(best_model, Config.MODEL_FULL_PATH)
    print(f"Full model saved to {Config.MODEL_FULL_PATH}")
    
    return model

# Train the model
model = train_model(model, dataloaders, criterion, optimizer, scheduler, Config.EPOCHS, Config.PATIENCE)