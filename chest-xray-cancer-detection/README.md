# Chest X-Ray Cancer & Pneumonia Detection

An end-to-end deep learning project for detecting **Pneumonia and Lung Cancer** from chest X-ray images using PyTorch + Transfer Learning, with an AI-powered explanation layer via Claude API.

![Python](https://img.shields.io/badge/Python-3.9+-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red) ![Streamlit](https://img.shields.io/badge/Streamlit-UI-green) ![License](https://img.shields.io/badge/License-MIT-yellow)

---

##  Project Overview

| Feature | Details |
|---|---|
| **Task** | Multi-class classification (Normal / Pneumonia / Cancer) |
| **Model** | EfficientNetB3 (Transfer Learning) |
| **Dataset** | Chest X-Ray Images (Kaggle) ~5k images |
| **UI** | Streamlit web app |
| **AI Explainer** | Claude API — generates plain-language diagnosis reports |

---

## 🗂️ Project Structure

```
chest-xray-cancer-detection/
├── data/
│   └── download_dataset.py       # Auto-download from Kaggle
├── src/
│   ├── dataset.py                # Dataset loader & augmentations
│   ├── model.py                  # EfficientNetB3 model definition
│   ├── train.py                  # Training loop
│   ├── evaluate.py               # Metrics, confusion matrix, Grad-CAM
│   └── predict.py                # Single image inference
├── app/
│   └── streamlit_app.py          # Streamlit UI with Claude integration
├── notebooks/
│   └── 01_EDA.ipynb              # Exploratory Data Analysis
├── models/                       # Saved model checkpoints
├── tests/
│   └── test_model.py             # Unit tests
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/chest-xray-cancer-detection.git
cd chest-xray-cancer-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
```bash
cp .env.example .env
# Edit .env and add your Kaggle credentials and Claude API key
```

### 4. Download the dataset
```bash
python data/download_dataset.py
```

### 5. Train the model
```bash
python src/train.py --epochs 20 --batch_size 32 --lr 0.001
```

### 6. Evaluate
```bash
python src/evaluate.py --model_path models/best_model.pth
```

### 7. Launch the web app
```bash
streamlit run app/streamlit_app.py
```

---

## Dataset

**Source**: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) on Kaggle

```
data/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

---

## Model Architecture

- **Backbone**: EfficientNetB3 (pretrained on ImageNet)
- **Head**: Custom classifier (Dropout → Linear → BatchNorm → ReLU → Linear)
- **Loss**: CrossEntropyLoss with class weights
- **Optimizer**: AdamW with CosineAnnealingLR scheduler
- **Augmentations**: RandomHorizontalFlip, RandomRotation, ColorJitter, Normalize

---

## Claude API Integration

The app uses Claude to:
- Generate a **plain-language radiology-style report** from model predictions
- Explain **confidence levels** in patient-friendly terms
- Suggest **next steps** based on classification


## Disclaimer

This project is for **educational and research purposes only**. It is NOT a medical device and should NOT be used for actual clinical diagnosis.


