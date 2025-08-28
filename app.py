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

from gradcam import GradCAM, ViTGradCAM, apply_colormap_on_image
import cv2


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

def load_brain_tumor_model():
    model = load_model("brain_tumor_detection_model.h5")  
    return model

# Initialize models
monkeypox_model = None
kidney_model = None
brain_tumor_model = None
monkeypox_class_names = ['Monkeypox', 'Others']
kidney_class_names = ['Cyst', 'Normal', 'Stone', 'Tumor']  
brain_tumor_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

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

try:
    brain_tumor_model = load_brain_tumor_model()
    print("Brain Tumor model loaded successfully!")
except Exception as e:
    print(f"Error loading Brain Tumor model: {str(e)}")


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def generate_gradcam_keras(model, img_array, class_idx):
    """Generate GradCAM for Keras models with improved error handling"""
    try:
        # Print model summary for debugging
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        
        # Find the last convolutional layer
        last_conv_layer = None
        conv_layer_name = None
        
        for i, layer in enumerate(reversed(model.layers)):
            if 'conv' in layer.__class__.__name__.lower():
                last_conv_layer = layer
                conv_layer_name = layer.name
                print(f"Found convolutional layer: {layer.name} at index {len(model.layers)-1-i}")
                break
        
        if last_conv_layer is None:
            print("No convolutional layer found. Available layers:")
            for i, layer in enumerate(model.layers):
                print(f"  Layer {i}: {layer.name} ({layer.__class__.__name__})")
            return None
        
        print(f"Using layer: {conv_layer_name} for GradCAM")
        print(f"Layer output shape: {last_conv_layer.output_shape}")
        
        # Create gradient model
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs], 
            outputs=[last_conv_layer.output, model.output]
        )
        
        # Convert input to tensor
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        print(f"Input tensor shape: {img_tensor.shape}")
        
        # Compute gradients
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            conv_outputs, predictions = grad_model(img_tensor)
            loss = predictions[:, class_idx]
            print(f"Conv outputs shape: {conv_outputs.shape}")
            print(f"Predictions shape: {predictions.shape}")
            print(f"Loss shape: {loss.shape}")
        
        # Get gradients
        grads = tape.gradient(loss, conv_outputs)
        print(f"Gradients shape: {grads.shape}")
        
        if grads is None:
            print("Gradients are None - cannot compute GradCAM")
            return None
        
        # Handle different gradient dimensions
        if len(grads.shape) == 4:  # (batch, height, width, channels)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs_squeezed = conv_outputs[0]
        elif len(grads.shape) == 3:  # (height, width, channels) - no batch dimension
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
            conv_outputs_squeezed = conv_outputs
        else:
            print(f"Unexpected gradient shape: {grads.shape}")
            return None
        
        print(f"Pooled gradients shape: {pooled_grads.shape}")
        print(f"Conv outputs squeezed shape: {conv_outputs_squeezed.shape}")
        
        # Generate heatmap
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs_squeezed), axis=-1)
        print(f"Heatmap shape: {heatmap.shape}")
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        
        heatmap_np = heatmap.numpy()
        print(f"Final heatmap shape: {heatmap_np.shape}")
        print(f"Heatmap min/max: {heatmap_np.min():.3f}/{heatmap_np.max():.3f}")
        
        return heatmap_np
        
    except Exception as e:
        print(f"Error in GradCAM generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

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

            # ===== GradCAM =====
            heatmap_base64 = None
            try:
                # Use specialized ViT GradCAM for BViTNet models
                vit_gradcam = ViTGradCAM(monkeypox_model)
                heatmap = vit_gradcam.generate(img_tensor, predicted_class)
                if heatmap is not None:
                    overlay = apply_colormap_on_image(img, heatmap)
                    _, buffer = cv2.imencode('.png', overlay)
                    heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
                vit_gradcam.cleanup()
            except Exception as e:
                print(f"Error generating GradCAM for monkeypox: {str(e)}")
                import traceback
                traceback.print_exc()

            prediction = {
                'class': monkeypox_class_names[predicted_class],
                'probability': f"{probability:.2f}%",
                'is_monkeypox': predicted_class == 1,
                'heatmap': heatmap_base64
            }

            return jsonify(prediction)
        except Exception as e:
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

            # ===== GradCAM =====
            heatmap_base64 = None
            try:
                # Use standard GradCAM for VGG16 models
                gradcam = GradCAM(kidney_model, kidney_model.features[-1])
                heatmap = gradcam.generate(img_tensor, predicted_class)
                if heatmap is not None:
                    overlay = apply_colormap_on_image(img, heatmap)
                    _, buffer = cv2.imencode('.png', overlay)
                    heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
                gradcam.cleanup()
            except Exception as e:
                print(f"Error generating GradCAM for kidney: {str(e)}")
                import traceback
                traceback.print_exc()

            all_probs = {kidney_class_names[i]: f"{probabilities[i].item() * 100:.2f}%" 
                         for i in range(len(kidney_class_names))}

            prediction = {
                'class': kidney_class_names[predicted_class],
                'probability': f"{probability:.2f}%",
                'is_normal': predicted_class == 1,
                'all_probabilities': all_probs,
                'heatmap': heatmap_base64
            }

            return jsonify(prediction)
        except Exception as e:
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
            img_resized = img.resize((150, 150))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            predictions = brain_tumor_model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class] * 100

            # === Generate GradCAM for Keras model ===
            heatmap_base64 = None
            try:
                heatmap = generate_gradcam_keras(brain_tumor_model, img_array, predicted_class)
                if heatmap is not None:
                    # Resize heatmap to match original image
                    heatmap_resized = cv2.resize(heatmap, (150, 150))
                    overlay = apply_colormap_on_image(img_resized, heatmap_resized)
                    _, buffer = cv2.imencode('.png', overlay)
                    heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
                    print("Brain tumor GradCAM generated successfully")
                else:
                    print("Failed to generate heatmap - heatmap is None")
            except Exception as e:
                print(f"Error generating GradCAM for brain tumor: {str(e)}")
                import traceback
                traceback.print_exc()

            all_probs = {brain_tumor_labels[i]: f"{predictions[0][i] * 100:.2f}%" 
                         for i in range(len(brain_tumor_labels))}

            response = {
                'class': brain_tumor_labels[predicted_class],
                'confidence': f"{confidence:.2f}%",
                'all_probabilities': all_probs,
                'heatmap': heatmap_base64
            }

            return jsonify(response)

        except Exception as e:
            print(f"Error in brain tumor prediction: {str(e)}")
            return jsonify({'error': str(e)})

    return jsonify({'error': 'File type not allowed'})

# Keeping the original predict route for backward compatibility
@app.route('/predict', methods=['POST'])
def predict():
    return predict_monkeypox()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
