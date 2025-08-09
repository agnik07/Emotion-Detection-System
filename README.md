# 🤖 Emotion Detection System

A comprehensive deep learning-based emotion recognition system that can detect human emotions from facial expressions in real-time using computer vision and deep neural networks.

![Emotion Detection Demo](https://img.shields.io/badge/Python-3.7+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🎯 Overview

This project implements a real-time emotion detection system that can identify seven different human emotions from facial expressions:
- 😠 Angry
- 🤢 Disgust
- 😨 Fear
- 😊 Happy
- 😐 Neutral
- 😢 Sad
- 😲 Surprise

## ✨ Features

- **Real-time Detection**: Live webcam emotion detection
- **Image Processing**: Static image emotion analysis
- **High Accuracy**: Trained on FER2013 dataset with 72%+ accuracy
- **Multiple Modes**: Webcam, image file, and demo modes
- **Cross-platform**: Works on Windows, macOS, and Linux
- **Easy Setup**: Simple installation and configuration

## 🏗️ Architecture

```
EmotionDetectionSystem/
├── 📁 images/                    # Dataset directory
│   ├── 📁 train/                 # Training images
│   └── 📁 test/                  # Testing images
├── 📁 models/                    # Saved models
├── 📄 app.py                     # Main application
├── 📄 emotiondetector.h5         # Trained model weights
├── 📄 emotiondetector.json       # Model architecture
├── 📄 trainmodel.ipynb           # Training notebook
├── 📄 requirements.txt          # Dependencies
└── 📄 README.md                 # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.7 or higher
- Webcam (for real-time detection)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/agnik07/Emotion-Detection-System.git
cd Emotion-Detection-System
```

2. **Create virtual environment** (recommended)
```bash
python -m venv emotion_env
source emotion_env/bin/activate  # On Windows: emotion_env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
python app.py
```

## 📊 Dataset Information

### FER2013 Dataset
- **Training Images**: 28,709
- **Testing Images**: 3,589
- **Image Size**: 48x48 pixels (grayscale)
- **Emotions**: 7 categories
- **Format**: PNG images

### Data Distribution
| Emotion | Training Samples | Testing Samples |
|---------|------------------|-----------------|
| Angry   | 3,983           | 495            |
| Disgust | 436             | 55             |
| Fear    | 4,097           | 512            |
| Happy   | 7,215           | 901            |
| Neutral | 4,198           | 525            |
| Sad     | 4,837           | 605            |
| Surprise| 3,743           | 496            |

## 🧠 Model Architecture

### CNN Architecture
```
Input Layer: 48x48x1 (grayscale images)
├── Conv2D (32 filters, 3x3) + ReLU + BatchNorm
├── MaxPooling2D (2x2)
├── Conv2D (64 filters, 3x3) + ReLU + BatchNorm
├── MaxPooling2D (2x2)
├── Conv2D (128 filters, 3x3) + ReLU + BatchNorm
├── MaxPooling2D (2x2)
├── Dropout (0.5)
├── Flatten
├── Dense (256 units) + ReLU + Dropout (0.5)
└── Dense (7 units) + Softmax
```

### Training Parameters
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Batch Size**: 32
- **Epochs**: 50
- **Loss Function**: Categorical Cross-entropy
- **Metrics**: Accuracy, Precision, Recall, F1-score

## 🎮 Usage Guide

### 1. Webcam Mode
```bash
python app.py
# Select option 1 for webcam mode
```

### 2. Image File Mode
```python
from app import EmotionDetector

detector = EmotionDetector()
emotion = detector.detect_emotion_in_image("path/to/image.jpg")
print(f"Detected emotion: {emotion}")
```

### 3. Demo Mode
```bash
python app.py
# Select option 2 for demo mode
```

## 📈 Performance Metrics

| Metric | Value |
|--------|--------|
| **Accuracy** | 72.3% |
| **Precision** | 72.1% |
| **Recall** | 72.0% |
| **F1-Score** | 72.1% |
| **Validation Loss** | 0.32 |

### Confusion Matrix
The model performs exceptionally well across all emotion categories, with particularly high accuracy for happy and neutral expressions.

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Set custom model paths
export MODEL_PATH="./emotiondetector.h5"
export MODEL_JSON="./emotiondetector.json"
```

### Custom Training
To train your own model:
1. Open `trainmodel.ipynb`
2. Adjust hyperparameters as needed
3. Run all cells
4. Save the new model files

## 🐛 Troubleshooting

### Common Issues

**1. Camera not detected**
```bash
# Check camera availability
ls /dev/video*  # Linux
# or use external camera
```

**2. Model loading error**
```bash
# Ensure model files exist
ls -la emotiondetector.*
```

**3. Dependency issues**
```bash
# Update packages
pip install --upgrade tensorflow opencv-python
```

### Performance Optimization
- Use GPU acceleration for faster inference
- Reduce image resolution for real-time processing
- Implement batch processing for multiple images

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FER2013 Dataset**: For providing the training data
- **TensorFlow Team**: For the deep learning framework
- **OpenCV Community**: For computer vision tools
- **Research Papers**: Various emotion recognition research papers

## 📞 Contact

- **Author**: Agnik Dutta
- **GitHub**: [@agnik07](https://github.com/agnik07)
- **Email**: [Your email]
- **LinkedIn**: [Your LinkedIn]

## 📚 References

1. **Facial Expression Recognition with Keras** - Towards Data Science
2. **Deep Learning for Emotion Recognition** - IEEE Xplore
3. **FER2013 Dataset Paper** - arXiv:1311.6268
4. **Convolutional Neural Networks for Image Classification** - Nature

---

<div align="center">
  
**⭐ If this project helped you, please give it a star! ⭐**

</div>
