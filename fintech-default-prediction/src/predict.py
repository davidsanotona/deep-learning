import torch
import numpy as np
import os
import joblib
from src.model import DefaultPredictorNN

def predict_default(raw_applicant_features, input_dim):
    """
    Takes an array of an applicant's RAW features, scales them, and returns default probability.
    """
    model_path = os.path.join('models', 'default_predictor.pth')
    scaler_path = os.path.join('models', 'scaler.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Model or Scaler not found. Please run --train first.")


    scaler = joblib.load(scaler_path)
    
    raw_array = np.array(raw_applicant_features).reshape(1, -1)
    scaled_features = scaler.transform(raw_array)

    model = DefaultPredictorNN(input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        X_tensor = torch.tensor(scaled_features, dtype=torch.float32)
        logits = model(X_tensor)
        probability = torch.sigmoid(logits).item()
        
    return probability