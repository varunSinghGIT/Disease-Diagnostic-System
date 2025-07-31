# app.py - Main 
import os
import torch
import torch.nn as nn
from flask import Flask, request, render_template, jsonify
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import vgg16
from transformers import ViTModel
from werkzeug.utils import secure_filename
import io
import base64
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

# Set up Flask app
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}


os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define model classes for monkeypox detection
class BroadAttention(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(BroadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, queries, keys, values):
        queries = torch.cat(queries, dim=1)
        keys = torch.cat(keys, dim=1)
        values = torch.cat(values, dim=1)
        attention_output, _ = self.attention(queries, keys, values)
        pooled_output = self.pool(attention_output.permute(0, 2, 1)).squeeze(-1)
        return pooled_output

class BViTNet(nn.Module):
    def __init__(self, vit_model_name, num_classes, gamma=0.5):
        super(BViTNet, self).__init__()
        self.vit = ViTModel.from_pretrained(vit_model_name, output_attentions=True)
        self.hidden_size = self.vit.config.hidden_size
        self.num_layers = len(self.vit.encoder.layer)
        self.gamma = gamma
        self.broad_attention = BroadAttention(self.hidden_size, self.num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        deep_features = hidden_states[-1][:, 0, :]
        queries = [layer[:, 0, :].unsqueeze(1) for layer in hidden_states]
        broad_features = self.broad_attention(queries, queries, queries)
        combined_features = deep_features + self.gamma * broad_features
        return self.classifier(combined_features)

# Define VGG16 model for kidney disease detection 
class KidneyVGG16(nn.Module):
    def __init__(self, num_classes=4):
        super(KidneyVGG16, self).__init__()
        # Load pre-trained VGG16 model
        vgg_model = vgg16(pretrained=True)
        # Extract features from VGG16
        self.features = vgg_model.features
        # Define classifier exactly 
        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Image preprocessing for monkeypox model
monkeypox_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Image preprocessing for kidney model 
kidney_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

brain_tumor_transform = transforms.Compose([
    transforms.Resize((150, 150)),  # Match model input size
    transforms.ToTensor(),
])

# Load monkeypox model
def load_monkeypox_model():
    model = BViTNet("google/vit-base-patch16-224-in21k", 2, 0.5)
    # Load the trained weights
    model.load_state_dict(torch.load('model.pth', map_location=device))
    model.to(device)
    model.eval()
    return model

# Load kidney disease model
def load_kidney_model():
    model = KidneyVGG16(num_classes=4)  # 4 classes 
    # Load the trained weights 
    model.load_state_dict(torch.load('kidney_model.pth', map_location=device))
    model.to(device)
    model.eval()
    return model

# Initialize models
monkeypox_model = None
kidney_model = None
monkeypox_class_names = ['Monkeypox', 'Others']
kidney_class_names = ['Cyst', 'Normal', 'Stone', 'Tumor']  

# Initialize models at startup
try:
    monkeypox_model = load_monkeypox_model()
    print("Monkeypox model loaded successfully!")
except Exception as e:
    print(f"Error loading monkeypox model: {str(e)}")

try:
    kidney_model = load_kidney_model()
    print("Kidney disease model loaded successfully!")
except Exception as e:
    print(f"Error loading kidney model: {str(e)}")

def load_brain_tumor_model():
    model = load_model("brain_tumor_detection_model.h5")  
    return model

brain_tumor_model = None
brain_tumor_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

try:
    brain_tumor_model = load_brain_tumor_model()
    print("Brain Tumor model loaded successfully!")
except Exception as e:
    print(f"Error loading Brain Tumor model: {str(e)}")


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_monkeypox', methods=['POST'])
def predict_monkeypox():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        try:
            img = Image.open(file.stream).convert('RGB')
            img_tensor = monkeypox_transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = monkeypox_model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                predicted_class = torch.argmax(probabilities).item()
                probability = probabilities[predicted_class].item() * 100

                # Debugging: Print the predicted class and probability
                print(f"Predicted Class Index: {predicted_class}")  
                print(f"Predicted Class: {monkeypox_class_names[predicted_class]}")
                print(f"Probability: {probability:.2f}%")

                prediction = {
                    'class': monkeypox_class_names[predicted_class],
                    'probability': f"{probability:.2f}%",
                    'is_monkeypox': predicted_class == 1
                }

                return jsonify(prediction)
        except Exception as e:
            print(f"Error: {str(e)}")  # Debugging error messages
            return jsonify({'error': str(e)})
    
    return jsonify({'error': 'File type not allowed'})

@app.route('/predict_kidney', methods=['POST'])
def predict_kidney():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        try:
            img = Image.open(file.stream).convert('RGB')
            img_tensor = kidney_transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = kidney_model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                predicted_class = torch.argmax(probabilities).item()
                probability = probabilities[predicted_class].item() * 100

                # Get all class probabilities for display
                all_probs = {kidney_class_names[i]: f"{probabilities[i].item() * 100:.2f}%" 
                            for i in range(len(kidney_class_names))}

                # Debugging: Print the predicted class and probability
                print(f"Kidney Prediction - Class Index: {predicted_class}")  
                print(f"Kidney Prediction - Class: {kidney_class_names[predicted_class]}")
                print(f"Kidney Prediction - Probability: {probability:.2f}%")

                prediction = {
                    'class': kidney_class_names[predicted_class],
                    'probability': f"{probability:.2f}%",
                    'is_normal': predicted_class == 1,  # Index 1 is 'Normal' in your class list
                    'all_probabilities': all_probs  # Include all class probabilities
                }

                return jsonify(prediction)
        except Exception as e:
            print(f"Error: {str(e)}")  # Debugging error messages
            return jsonify({'error': str(e)})
    
    return jsonify({'error': 'File type not allowed'})

@app.route('/predict_brain_tumor', methods=['POST'])
def predict_brain_tumor():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        try:
            # Preprocess image
            img = Image.open(file.stream).convert('RGB')
            img = img.resize((150, 150))  # Match input size of model
            img_array = np.array(img) / 255.0  # Normalize
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Predict
            predictions = brain_tumor_model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class] * 100

            # Get all probabilities
            all_probs = {
                brain_tumor_labels[i]: f"{predictions[0][i] * 100:.2f}%"
                for i in range(len(brain_tumor_labels))
            }

            response = {
                'class': brain_tumor_labels[predicted_class],
                'confidence': f"{confidence:.2f}%",
                'all_probabilities': all_probs
            }

            return jsonify(response)

        except Exception as e:
            return jsonify({'error': str(e)})

    return jsonify({'error': 'File type not allowed'})

# Keeping the original predict route for backward compatibility
@app.route('/predict', methods=['POST'])
def predict():
    return predict_monkeypox()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)