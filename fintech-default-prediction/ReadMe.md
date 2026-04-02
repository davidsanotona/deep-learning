# Fintech Loan Default Prediction
pytorch based deep learning model deisgned to predict default

The model is a Deep Feedforward Neural Network built with PyTorch, utilizing:
* **Batch Normalization** for training stability.
* **Dropout** layers to prevent overfitting on minority classes.
* **BCEWithLogitsLoss** for robust binary classification.

##  Dataset Download Instructions
Due to GitHub's file size limits, the raw dataset is not included in this repository. You must download it manually before running the code.

1. **Get the Data:** Visit the [Home Credit Default Risk Kaggle Page](https://www.kaggle.com/c/home-credit-default-risk/data)
2. **Download:** Click the black **"Download All"** button at the bottom of the page (requires a free Kaggle account).
3. **Extract:** Unzip the downloaded file and place all the resulting `.csv` files directly into the `data/raw/` folder in this project.

## Project Structure

fintech-default-prediction/
│
├── data/
│   └── raw/                 
│       ├── application_train.csv   
│       └── [other downloaded CSVs]
│
├── models/                  
│
├── src/
│   ├── __init__.py          
│   ├── data_loader.py       
│   ├── model.py             
│   ├── train.py             
│   └── predict.py           
│
├── main.py                 
├── requirements.txt         
└── README.md
## Quick Start
1. Train the model: python main.py --train

2. Test a prediction: python main.py --predict (need to be trained first)
