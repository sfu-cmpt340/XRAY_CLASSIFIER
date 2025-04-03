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
from torchvision import models
from tqdm import tqdm 


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

train_data_dir = "prepared_dataset/train"
val_data_dir = "prepared_dataset/val"
test_data_dir = "prepared_dataset/test"
batch_size = 32
learning_rate = 0.0001
num_epochs = 15
num_classes = 4
best_model_path = "new_best_resnet50_model.pth"
full_model_path = "new_best_resnet50_full_model.pth"
patience = 5
weight_decay = 1e-4

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(train_data_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(val_data_dir, transform=test_transforms)
test_dataset = datasets.ImageFolder(test_data_dir, transform=test_transforms)

data_loaders = {
    'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
    'valid': DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
    'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
}

def calc_class_weights(dataset):
    labels = [label for _, label in dataset.samples]
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    return torch.tensor(class_weights, dtype=torch.float).to(device)

train_class_weights = calc_class_weights(train_dataset)
loss_criterion = nn.CrossEntropyLoss(weight=train_class_weights)

def build_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    for param in list(model.parameters())[:-100]:
        param.requires_grad = False
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, num_classes)
    )
    return model.to(device)

model = build_model()

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

def train_model(model, data_loaders, loss_criterion, optimizer, lr_scheduler, num_epochs, patience):
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch = 0
    no_improve_epochs = 0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            progress_bar = tqdm(data_loaders[phase], desc=f"{phase.capitalize()} Epoch {epoch+1}/{num_epochs}", leave=False)
            
            for inputs, labels in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = loss_criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            epoch_acc = running_corrects / len(data_loaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'valid':
                lr_scheduler.step(epoch_loss)
                
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
        
        if no_improve_epochs >= patience:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break
            
        print()
    
    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f} at epoch {best_epoch+1}')
    
    model.load_state_dict(best_model_wts)
    
    torch.save({
        'epoch': best_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc
    }, best_model_path)
    
    torch.save(model, full_model_path)
    print(f"Full ResNet50 model saved to {full_model_path}")
    
    return model

model = train_model(model, data_loaders, loss_criterion, optimizer, lr_scheduler, num_epochs, patience)
