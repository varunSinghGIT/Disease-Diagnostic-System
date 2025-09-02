# diagnostics.py
import os
import torch
import torch.nn as nn
from flask import Blueprint, request, render_template, jsonify, current_app
from flask_login import login_required, current_user
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import vgg16
from transformers import ViTModel
import base64
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from models import db
from gradcam import GradCAM, ViTGradCAM, apply_colormap_on_image
import cv2
import requests
from dotenv import load_dotenv
import time
import random

# Load environment variables from a .env file
load_dotenv()

# --- OpenRouter API Configuration ---
API_KEY = os.getenv("OPENROUTER_API_KEY")
API_URL = 'https://openrouter.ai/api/v1/chat/completions'
PROMPT_TEMPLATE = """
You are an AI medical assistant. A user has received a potential diagnosis of {disease}.
The user has asked the following question: "{question}"

Please provide a comprehensive, helpful, and safe response. The response should be informative and easy to understand for a layperson. 
Focus on providing general information, prevention tips, and when to seek professional medical advice.

IMPORTANT: Start your response with a clear disclaimer that you are an AI assistant and your advice is not a substitute for professional medical consultation.
"""

# Set up Blueprint
diagnostics_bp = Blueprint('diagnostics', __name__)

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# --- Model Definitions ---

class BroadAttention(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(BroadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, queries, keys, values):
        queries = torch.cat(queries, dim=1)
        keys = torch.cat(keys, dim=1)
        values = torch.cat(values, dim=1)
        attention_output, _ = self.attention(queries, keys, values)
        return self.pool(attention_output.permute(0, 2, 1)).squeeze(-1)

class BViTNet(nn.Module):
    def __init__(self, vit_model_name, num_classes, gamma=0.5):
        super(BViTNet, self).__init__()
        self.vit = ViTModel.from_pretrained(vit_model_name, output_attentions=True)
        self.hidden_size = self.vit.config.hidden_size
        self.gamma = gamma
        self.broad_attention = BroadAttention(self.hidden_size, len(self.vit.encoder.layer))
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 512), nn.ReLU(), nn.Dropout(0.5),
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

class KidneyVGG16(nn.Module):
    def __init__(self, num_classes=4):
        super(KidneyVGG16, self).__init__()
        vgg_model = vgg16(pretrained=True)
        self.features = vgg_model.features
        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# --- Image Transformations ---
monkeypox_transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
kidney_transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# --- Model Loading ---
def load_monkeypox_model():
    model = BViTNet("google/vit-base-patch16-224-in21k", 2, 0.5)
    model.load_state_dict(torch.load('model.pth', map_location=device))
    return model.to(device).eval()

def load_kidney_model():
    model = KidneyVGG16(num_classes=4)
    model.load_state_dict(torch.load('kidney_model.pth', map_location=device))
    return model.to(device).eval()

def load_brain_tumor_model():
    return load_model("brain_tumor_detection_model.h5")

monkeypox_model, kidney_model, brain_tumor_model = None, None, None
monkeypox_class_names = ['Others', 'Monkeypox']
kidney_class_names = ['Cyst', 'Normal', 'Stone', 'Tumor']
brain_tumor_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

try: monkeypox_model = load_monkeypox_model(); print("Monkeypox model loaded.")
except Exception as e: print(f"Error loading monkeypox model: {e}")
try: kidney_model = load_kidney_model(); print("Kidney model loaded.")
except Exception as e: print(f"Error loading kidney model: {e}")
try: brain_tumor_model = load_brain_tumor_model(); print("Brain tumor model loaded.")
except Exception as e: print(f"Error loading brain tumor model: {e}")

# --- Utility Functions ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

def check_and_deduct_credit():
    if current_user.credits <= 0:
        return False, jsonify({'error': 'Insufficient credits. Please top up.'})
    current_user.credits -= 1
    db.session.commit()
    return True, None

# --- Routes ---
@diagnostics_bp.route('/')
@login_required
def index():
    return render_template('index.html')

@diagnostics_bp.route('/predict_monkeypox', methods=['POST'])
@login_required
def predict_monkeypox():
    can_proceed, response = check_and_deduct_credit()
    if not can_proceed: return response
    file = request.files.get('file')
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'})
    try:
        img = Image.open(file.stream).convert('RGB')
        img_tensor = monkeypox_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = monkeypox_model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            pred_class = torch.argmax(probs).item()
        
        # Grad-CAM
        heatmap_b64 = None
        try:
            cam = ViTGradCAM(monkeypox_model)
            heatmap = cam.generate(img_tensor, pred_class)
            overlay = apply_colormap_on_image(img, heatmap)
            _, buffer = cv2.imencode('.png', overlay)
            heatmap_b64 = base64.b64encode(buffer).decode('utf-8')
            cam.cleanup()
        except Exception as e: print(f"GradCAM error for monkeypox: {e}")
            
        return jsonify({
            'class': monkeypox_class_names[pred_class],
            'probability': f"{probs[pred_class].item()*100:.2f}",
            'is_monkeypox': pred_class == 1, # Assuming 'Monkeypox' is class 0
            'heatmap': heatmap_b64
        })
    except Exception as e: return jsonify({'error': str(e)})

@diagnostics_bp.route('/predict_kidney', methods=['POST'])
@login_required
def predict_kidney():
    can_proceed, response = check_and_deduct_credit()
    if not can_proceed: return response
    file = request.files.get('file')
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'})
    try:
        img = Image.open(file.stream).convert('RGB')
        img_tensor = kidney_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = kidney_model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            pred_class = torch.argmax(probs).item()
        
        all_probs = {kidney_class_names[i]: f"{probs[i].item()*100:.2f}" for i in range(len(kidney_class_names))}
        
        heatmap_b64 = None
        try:
            cam = GradCAM(kidney_model, kidney_model.features[-1])
            heatmap = cam.generate(img_tensor, pred_class)
            overlay = apply_colormap_on_image(img, heatmap)
            _, buffer = cv2.imencode('.png', overlay)
            heatmap_b64 = base64.b64encode(buffer).decode('utf-8')
            cam.cleanup()
        except Exception as e: print(f"GradCAM error for kidney: {e}")

        return jsonify({
            'class': kidney_class_names[pred_class],
            'all_probabilities': all_probs,
            'heatmap': heatmap_b64
        })
    except Exception as e: return jsonify({'error': str(e)})


@diagnostics_bp.route('/predict_brain_tumor', methods=['POST'])
@login_required
def predict_brain_tumor():
    can_proceed, response = check_and_deduct_credit()
    if not can_proceed: return response
    file = request.files.get('file')
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'})
    try:
        img = Image.open(file.stream).convert('RGB').resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        preds = brain_tumor_model.predict(img_array)[0]
        pred_class = np.argmax(preds)
        
        all_probs = {brain_tumor_labels[i]: f"{preds[i]*100:.2f}" for i in range(len(brain_tumor_labels))}

        return jsonify({
            'class': brain_tumor_labels[pred_class],
            'all_probabilities': all_probs,
            'heatmap': None # Grad-CAM for Keras model needs separate implementation
        })
    except Exception as e: return jsonify({'error': str(e)})

@diagnostics_bp.route('/chat', methods=['POST'])
@login_required
def chat():
    """Handles chat requests to the OpenRouter API with retry logic."""
    data = request.get_json()
    disease = data.get('disease')
    question = data.get('question')

    if not all([disease, question]):
        return jsonify({'error': 'Missing disease or question in request.'}), 400

    if not API_KEY:
        return jsonify({'error': 'API key for the chatbot is not configured on the server.'}), 500

    prompt = PROMPT_TEMPLATE.format(disease=disease, question=question)

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:5000", # Recommended by OpenRouter
        "X-Title": "AI Disease Diagnostic System" # Recommended by OpenRouter
    }
    payload = {
        "model": "deepseek/deepseek-chat-v3-0324:free",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 1024
    }

    max_retries = 4
    base_wait_time = 1  # seconds

    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()  # Raise an exception for 4xx/5xx errors
            
            response_data = response.json()
            if 'choices' in response_data and response_data['choices']:
                content = response_data['choices'][0]['message']['content']
                return jsonify({'response': content})
            else:
                return jsonify({'error': 'Received an invalid response from the AI service.'}), 500

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                if attempt < max_retries - 1:
                    wait_time = (base_wait_time * 2 ** attempt) + random.uniform(0, 1)
                    print(f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                else:
                    print("All retries failed. The API is still busy.")
                    return jsonify({'error': 'The AI service is currently busy. Please try again in a few moments.'}), 429
            else:
                print(f"An HTTP error occurred: {e}")
                return jsonify({'error': f'An HTTP error occurred: {e.response.status_code}'}), e.response.status_code
        
        except requests.exceptions.RequestException as e:
            print(f"A network error occurred: {e}")
            return jsonify({'error': 'Failed to connect to the AI service. Please check your network connection.'}), 500
            
    return jsonify({'error': 'Failed to get a response after multiple attempts.'}), 500
