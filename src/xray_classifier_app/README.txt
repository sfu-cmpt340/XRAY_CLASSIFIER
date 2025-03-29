to install all the dependencies run this command in your terminal:
pip install numpy flask tensorflow torch torchvision Pillow



Paste the pre-trained models (cnn_model.h5, resnet50_chest_diagnosis.h5, dense_net.pth) in the xray_classifier_app folder in such a way:

xray_classifier_app/
├── app.py
├── cnn_model.h5
├── resnet50_chest_diagnosis.h5
├── dense_net.pth
├── uploads/   
├── templates/
│   ├── landing.html
│   └── evaluate.html
└── static/s
    └── css/
         └── style.css