import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import roc_auc_score, classification_report

from src.data_loader import LoanDataset
from src.model import DefaultPredictorNN

def train_model():
    data_path =os.path.join('data', 'raw', 'application_train.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found at {data_path}. Please download it manually from Kaggle and place it in data/raw/")
        
    print("Loading Home Credit data into Pandas...")
    df = pd.read_csv(data_path)

    numeric_features = [
        'AMT_INCOME_TOTAL', 
        'AMT_CREDIT', 
        'AMT_ANNUITY', 
        'DAYS_BIRTH', 
        'DAYS_EMPLOYED',
        'EXT_SOURCE_2', 
        'EXT_SOURCE_3'
    ]

    # drop missing value
    df = df.dropna(subset=numeric_features + ['TARGET'])

    #split and scale
    X_raw = df[numeric_features].values
    y_raw = df['TARGET'].values
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, os.path.join('models', 'scaler.pkl'))

    # setup dataloader
    train_dataset = LoanDataset(X_train_scaled, y_train)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    # initialize model
    input_dim = X_train.shape[1]
    model = DefaultPredictorNN(input_dim)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    #train loop
    epochs = 10
    for epoch in range(epochs):
        model.train()
        epoch_loss=0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(train_loader):.4f}")
    #evaluate
    print("\nEvaluating model on 20% hold-out test set...")
    model.eval() # Turn off Dropout for testing
    
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

        test_logits = model(X_test_tensor)
        test_probs = torch.sigmoid(test_logits).squeeze().numpy()
        test_preds = (test_probs > 0.5).astype(int)

    auc_score = roc_auc_score(y_test, test_probs)
    print(f"ROC AUC Score: {auc_score:.4f}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, test_preds, zero_division=0)) 
    print("==========================================\n")
    # Save
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', 'default_predictor.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")