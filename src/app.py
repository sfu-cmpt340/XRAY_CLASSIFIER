import os
import json
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Set device for PyTorch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)

# -------------------------------
# 1) Define your class labels
# -------------------------------
class_names = {
    0: 'covid',
    1: 'normal',
    2: 'pneumonia',
    3: 'tuberculosis'
}

# -------------------------------
# 2) Load your models
# -------------------------------
# -- A) Keras model #1 (cnn_model.h5)
cnn_model = load_model('cnn_model.h5')

# -- B) Keras model #2 (resnet50_chest_diagnosis.h5)
resnet_model = load_model('resnet50_chest_diagnosis.h5')

# -- C) PyTorch model (densenet.pth)
try:
    # Load the entire model instead of just the state dictionary
    dense_net_model = torch.load("densenet.pth", map_location=DEVICE, weights_only=False)
    dense_net_model.to(DEVICE)
    dense_net_model.eval()
except Exception as e:
    dense_net_model = None
    print("ERROR loading DenseNet model. Make sure you have the correct file.")
    print(e)


# -- D) PyTorch VGG16 Model (vgg16.pth)
class XRayVGG16(nn.Module):
    def __init__(self, num_classes=4):
        super(XRayVGG16, self).__init__()
        base_vgg16 = models.vgg16(pretrained=False)
        for param in base_vgg16.features.parameters():
            param.requires_grad = False
        num_ftrs = base_vgg16.classifier[0].in_features
        base_vgg16.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        self.features = base_vgg16.features
        self.classifier = base_vgg16.classifier

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

try:
    vgg16_model = XRayVGG16(num_classes=4)
    vgg16_model.load_state_dict(torch.load("vgg16.pth", map_location=DEVICE, weights_only=False))
    vgg16_model.to(DEVICE)
    vgg16_model.eval()
    print("VGG16 custom model loaded successfully.")
except Exception as e:
    vgg16_custom_model = None
    print("ERROR loading VGG16 custom model.", e)

# -------------------------------
# 3) Define image transforms for PyTorch
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------------------
# 4) Helper functions for predictions and evaluation
# -------------------------------
def predict_with_keras(model, img_path):
    """Preprocess an image file and predict its label using a Keras model."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    preds = model.predict(img_array)
    predicted_index = np.argmax(preds, axis=1)[0]
    return class_names.get(predicted_index, "Unknown")

def predict_with_torch(model, img_path):
    """Preprocess an image file and predict its label using a PyTorch model."""
    if model is None:
        return "Model not loaded"
    img = Image.open(img_path).convert('RGB')
    img_t = transform(img)
    img_t = img_t.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_t)
        _, predicted = torch.max(outputs, 1)
        predicted_index = predicted.item()
    return class_names.get(predicted_index, "Unknown")

def evaluate_model_on_test_folder(model_obj, model_type, folder_path="test"):
    """
    Evaluates a model on the given folder path.
    If the folder's base name matches one of the expected class labels,
    it is treated as a single-class folder. Otherwise, the folder is assumed
    to contain subfolders for each class.
    
    Returns a dictionary with counts: correct, incorrect, and total per class.
    """
    # Initialize results with zeros for all classes.
    results = {label: {'correct': 0, 'incorrect': 0, 'total': 0} for label in class_names.values()}
    
    base_name = os.path.basename(os.path.normpath(folder_path)).lower()
    expected_labels = [label.lower() for label in class_names.values()]
    
    if base_name in expected_labels:
        # Treat folder_path as a single class folder.
        label = base_name
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total = 0
        correct = 0
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            if model_type == 'keras':
                predicted_label = predict_with_keras(model_obj, img_path)
            else:
                predicted_label = predict_with_torch(model_obj, img_path)
            total += 1
            if predicted_label.lower() == label:
                correct += 1
        results[label] = {'correct': correct, 'incorrect': total - correct, 'total': total}
    else:
        # Assume folder_path contains subfolders for each class.
        for label in class_names.values():
            label_folder = os.path.join(folder_path, label)
            if not os.path.exists(label_folder):
                results[label] = {'correct': 0, 'incorrect': 0, 'total': 0}
                continue
            image_files = [f for f in os.listdir(label_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            total = 0
            correct = 0
            for img_file in image_files:
                img_path = os.path.join(label_folder, img_file)
                if model_type == 'keras':
                    predicted_label = predict_with_keras(model_obj, img_path)
                else:
                    predicted_label = predict_with_torch(model_obj, img_path)
                total += 1
                if predicted_label.lower() == label.lower():
                    correct += 1
            results[label] = {'correct': correct, 'incorrect': total - correct, 'total': total}
    return results

# -------------------------------
# 5) Routes
# -------------------------------

# Landing page: choose a model.
@app.route("/")
def landing():
    return render_template("landing.html")

# Evaluation page: the user specifies the folder path on the server and/or uploads a single image.
@app.route("/evaluate", methods=["GET", "POST"])
def evaluate():
    # Get selected model key from query string or form data.
    model_key = request.args.get("model") or request.form.get("model")
    models_dict = {
        "cnn_model": {"object": cnn_model, "type": "keras", "name": "CNN Model"},
        "resnet_model": {"object": resnet_model, "type": "keras", "name": "ResNet Model"},
        "dense_net_model": {"object": dense_net_model, "type": "torch", "name": "DenseNet Model"},
        "vgg_model": {"object": vgg16_model, "type": "torch", "name": "VGG Model"}  # Added VGG model
    }
    if model_key not in models_dict:
        return redirect(url_for("landing"))
    selected_model = models_dict[model_key]["object"]
    model_type = models_dict[model_key]["type"]
    model_name = models_dict[model_key]["name"]

    folder_results = None
    single_prediction = None
    # Default folder path is "test"
    folder_path = request.form.get("folder_path", "test")

    if request.method == "POST":
        # Process folder evaluation if folder_submit button pressed.
        if "folder_submit" in request.form:
            folder_results = evaluate_model_on_test_folder(selected_model, model_type, folder_path)
        # Process single image upload.
        elif "single_file" in request.files and request.files["single_file"].filename.strip() != "":
            file = request.files["single_file"]
            upload_folder = "uploads"
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)
            filepath = os.path.join(upload_folder, file.filename)
            file.save(filepath)
            if model_type == "keras":
                single_prediction = predict_with_keras(selected_model, filepath)
            else:
                single_prediction = predict_with_torch(selected_model, filepath)
            os.remove(filepath)
    return render_template("evaluate.html",
                           model_name=model_name,
                           model_key=model_key,
                           folder_path=folder_path,
                           folder_results=folder_results,
                           single_prediction=single_prediction)

if __name__ == '__main__':
    app.run(debug=True)