import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models

# ===============================
# 1. Check & Enable GPU (Optional)
# ===============================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == 'cpu':
    print("No GPU detected; training will run on CPU.")

# ============================
# 2. Set Up Directory Paths
# ============================
base_dir = r"C:\Users\Jawdat Eissa\Desktop\Project\VGG16_model_training\preprocessed_data"
train_dir = os.path.join(base_dir, "train")
val_dir   = os.path.join(base_dir, "val")
test_dir  = os.path.join(base_dir, "test")

# ================================
# 3. Data Transforms & Datasets
# ================================
img_size = 224  # VGG16 default input size

data_transforms = {
    'train': transforms.Compose([
         transforms.Resize((256, 256)),
         transforms.RandomResizedCrop(img_size),
         transforms.RandomRotation(20),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], 
                              [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
         transforms.Resize((img_size, img_size)),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], 
                              [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
         transforms.Resize((img_size, img_size)),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], 
                              [0.229, 0.224, 0.225])
    ]),
}

datasets_dict = {
    'train': datasets.ImageFolder(train_dir, data_transforms['train']),
    'val': datasets.ImageFolder(val_dir, data_transforms['val']),
    'test': datasets.ImageFolder(test_dir, data_transforms['test'])
}

batch_size = 32
dataloaders = {
    'train': torch.utils.data.DataLoader(datasets_dict['train'], batch_size=batch_size, shuffle=True, num_workers=4),
    'val': torch.utils.data.DataLoader(datasets_dict['val'], batch_size=batch_size, shuffle=True, num_workers=4),
    'test': torch.utils.data.DataLoader(datasets_dict['test'], batch_size=batch_size, shuffle=False, num_workers=4)
}

dataset_sizes = {x: len(datasets_dict[x]) for x in ['train', 'val', 'test']}
class_names = datasets_dict['train'].classes

# =================================
# 4. Load Pretrained VGG16 Model
# =================================
model_ft = models.vgg16(pretrained=True)
# Freeze the feature extractor parameters
for param in model_ft.features.parameters():
    param.requires_grad = False

# ============================
# 5. Modify the Classifier
# ============================
num_ftrs = model_ft.classifier[0].in_features
model_ft.classifier = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(256, 4)  # 4 classes
)
model_ft = model_ft.to(device)

# ============================
# 6. Set Up Loss and Optimizer
# ============================
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.Adam(model_ft.classifier.parameters(), lr=1e-4)
# Optional: set up a learning rate scheduler
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# =====================================
# 7. Training Function Definition
# =====================================
def train_model(model, criterion, optimizer, scheduler, num_epochs=5):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluation mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward pass; track gradients only if in train phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if validation accuracy improves
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

def main():
    # ===============================
    # 8. Initial Training (Frozen)
    # ===============================
    print("Starting initial training with frozen feature extractor...")
    global model_ft, optimizer_ft, exp_lr_scheduler  # use globals to update these variables
    model_trained = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=5)

    # =======================================
    # 9. Fine-Tuning (Unfreeze Some Layers)
    # =======================================
    # Unfreeze the last few layers of the feature extractor
    fine_tune_at = 15  # example index
    for i, layer in enumerate(model_trained.features):
        if i >= fine_tune_at:
            for param in layer.parameters():
                param.requires_grad = True

    # Recreate optimizer to include now trainable parameters
    optimizer_ft_new = optim.Adam(filter(lambda p: p.requires_grad, model_trained.parameters()), lr=1e-5)
    # Optionally, you can reinitialize the scheduler here if needed
    exp_lr_scheduler_new = lr_scheduler.StepLR(optimizer_ft_new, step_size=7, gamma=0.1)

    print("Starting fine-tuning...")
    model_trained = train_model(model_trained, criterion, optimizer_ft_new, exp_lr_scheduler_new, num_epochs=10)

    # ===============================
    # 10. Evaluate Model on Test Set
    # ===============================
    model_trained.eval()
    running_corrects = 0
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model_trained(inputs)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)

    test_acc = running_corrects.double() / dataset_sizes['test']
    print(f"\nTest Accuracy: {test_acc:.4f}")

    # ===============================
    # 11. Save Final Model
    # ===============================
    # Save the entire model, including its architecture and weights.
    torch.save(model_trained, "vgg16_custom_finetuned_full.pth")
    print("Final model saved as vgg16_custom_finetuned.pth")
    print("Training script completed successfully.")

if __name__ == '__main__':
    main()
