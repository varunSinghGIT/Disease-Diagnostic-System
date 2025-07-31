# Disease-Diagnostic-System
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://tensorflow.org/)

A comprehensive web-based medical image analysis system that leverages deep learning to assist in the diagnosis of multiple diseases including Monkeypox, Kidney diseases, and Brain tumors. The system provides an intuitive interface with real-time analysis capabilities and an integrated AI chatbot for medical guidance.

## 🎯 Features

### Multi-Disease Detection
- **Monkeypox Detection**: Advanced BViT (Broad Vision Transformer) model for skin lesion analysis
- **Kidney Disease Analysis**: VGG16-based classification for Cyst, Normal, Stone, and Tumor detection
- **Brain Tumor Detection**: CNN model for identifying Glioma, Meningioma, Pituitary tumors, or Normal brain tissue

### User Experience
- Clean, responsive web interface with Bootstrap 5
- Real-time image preview and analysis
- Progress bars showing confidence levels for all classes
- Tabbed interface for easy navigation between different diagnostic tools
- Mobile-friendly design

### AI Assistant
- Integrated chatbot powered by TinyLlama model
- Medical guidance and information about diagnostic results
- Context-aware responses for medical imaging questions

## 🏗️ System Architecture

```
disease-diagnostic-system/
├── app.py                          # Main Flask application
├── chatapp.py                      # Chatbot functionality
├── templates/
│   └── index.html                  # Main web interface
├── models/
│   ├── model.pth                   # Monkeypox BViT model
│   ├── kidney_model.pth            # Kidney VGG16 model
│   └── brain_tumor_detection_model.h5  # Brain tumor CNN model
├── uploads/                        # Image upload directory
├── notebook/                       # Jupyter notebooks for training
│   ├── Monkeypox-Bvit.ipynb      # Monkeypox model training
│   ├── kidney-vgg16.ipynb         # Kidney model training
│   └── brain-tumor-detection-96-accuracy.ipynb
└── requirements.txt                # Python dependencies
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster inference)
- 8GB+ RAM recommended

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/disease-diagnostic-system.git
cd disease-diagnostic-system
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Pre-trained Models
Ensure the following model files are in the project root:
- `model.pth` - Monkeypox BViT model
- `kidney_model.pth` - Kidney VGG16 model  
- `brain_tumor_detection_model.h5` - Brain tumor CNN model

### Step 5: Configure Hugging Face Cache (Optional)
```bash
# Set custom cache directory for Hugging Face models
export HF_HOME=/path/to/your/cache/directory
```

## 🎮 Usage

### Starting the Application
```bash
python app.py
```

The application will be available at `http://localhost:5000`

### Using the Diagnostic Tools

#### 1. Monkeypox Detection
1. Navigate to the "Monkeypox Detection" tab
2. Upload a skin lesion image (PNG, JPG, JPEG)
3. Click "Analyze" to get results
4. View confidence scores and medical recommendations

#### 2. Kidney Analysis
1. Switch to the "Kidney Analysis" tab
2. Upload a kidney CT scan image
3. Get classification results for:
   - Normal kidney
   - Kidney cyst
   - Kidney stone
   - Kidney tumor

#### 3. Brain Tumor Detection
1. Go to the "Brain Tumor Detection" tab
2. Upload a brain MRI scan
3. Receive analysis for:
   - Glioma
   - Meningioma
   - Pituitary tumor
   - No tumor detected

### AI Chatbot
- Click the 🤖 button in the bottom-right corner
- Ask questions about medical imaging, diagnostic results, or general health information
- Get contextual responses from the AI assistant

## 🧠 Model Details

### Monkeypox Detection Model (BViTNet)
- **Architecture**: Broad Vision Transformer with attention mechanism
- **Base Model**: Google ViT-base-patch16-224-in21k
- **Classes**: Monkeypox, Others
- **Input Size**: 224x224 pixels
- **Preprocessing**: Normalization with mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]

### Kidney Disease Model (VGG16)
- **Architecture**: Modified VGG16 with custom classifier
- **Classes**: Cyst, Normal, Stone, Tumor
- **Input Size**: 224x224 pixels
- **Features**: 25,088 → 4,096 → 1,024 → 4 classes

### Brain Tumor Model (CNN)
- **Architecture**: Custom Convolutional Neural Network
- **Classes**: Glioma, Meningioma, No Tumor, Pituitary
- **Input Size**: 150x150 pixels
- **Accuracy**: 96%+ on test dataset

## 📊 API Endpoints

### Image Analysis Endpoints
```http
POST /predict_monkeypox
Content-Type: multipart/form-data
Body: file (image)

POST /predict_kidney  
Content-Type: multipart/form-data
Body: file (image)

POST /predict_brain_tumor
Content-Type: multipart/form-data  
Body: file (image)
```

### Chatbot Endpoint
```http
POST /chat
Content-Type: application/json
Body: {"message": "Your question here"}
```

### Response Format
```json
{
  "class": "predicted_class",
  "probability": "confidence_percentage",
  "all_probabilities": {
    "class1": "percentage1",
    "class2": "percentage2"
  }
}
```

## ⚙️ Configuration

### Environment Variables
```bash
# Hugging Face cache directory
HF_HOME=/path/to/cache

# Flask configuration
FLASK_ENV=development
FLASK_DEBUG=True
```

### Application Settings
- **Max Upload Size**: 16MB
- **Allowed Extensions**: PNG, JPG, JPEG
- **Upload Directory**: `uploads/`
- **Model Device**: Auto-detected (CUDA/CPU)

## 📋 Requirements

```txt
torch>=1.9.0
torchvision>=0.10.0
flask>=2.0.0
transformers>=4.20.0
tensorflow>=2.8.0
pillow>=8.3.0
numpy>=1.21.0
langchain-huggingface>=0.0.3
werkzeug>=2.0.0
```

## 🧪 Model Training

The project includes Jupyter notebooks for training each model:

### Training Monkeypox Model
```bash
jupyter notebook notebook/Monkeypox-Bvit.ipynb
```

### Training Kidney Model  
```bash
jupyter notebook notebook/kidney-vgg16.ipynb
```

### Training Brain Tumor Model
```bash
jupyter notebook notebook/brain-tumor-detection-96-accuracy.ipynb
```

## 🚨 Important Medical Disclaimer

**⚠️ MEDICAL DISCLAIMER**: This system is designed for educational and research purposes only. It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.

### Key Points:
- Results are probabilistic and may contain errors
- False positives and false negatives are possible
- Always seek professional medical consultation
- Do not delay seeking medical care based on these results
- This tool is meant to assist, not replace, medical professionals

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Include unit tests for new features
- Update documentation as needed

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Varun Kumar Singh**
- GitHub: [@varun-singh]
- Email: [your-email@example.com]

## 🙏 Acknowledgments

- Hugging Face for transformer models
- Google for Vision Transformer architecture
- PyTorch and TensorFlow communities
- Medical imaging datasets contributors
- Bootstrap team for UI components

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/disease-diagnostic-system/issues) page
2. Create a new issue with detailed description
3. Include error logs and system information
4. Tag with appropriate labels (bug, feature, question)

## 🔄 Version History

- **v1.0.0** - Initial release with three diagnostic models
- **v1.1.0** - Added AI chatbot integration
- **v1.2.0** - Improved UI/UX and mobile responsiveness

---

**⭐ If you find this project helpful, please consider giving it a star!**
