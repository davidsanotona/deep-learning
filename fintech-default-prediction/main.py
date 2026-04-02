import argparse
import numpy as np
import os
import joblib
from src.train import train_model
from src.predict import predict_default

def main():
    parser = argparse.ArgumentParser(description="Fintech Loan Default Prediction Pipeline")
    parser.add_argument('--train', action='store_true', help='Train the model and save weights/scaler')
    parser.add_argument('--predict', action='store_true', help='Run inference for a new applicant')

    args = parser.parse_args()

    if args.train:
        print("Initiating model training...")
        train_model()

    if args.predict:
        if not os.path.exists(os.path.join('models', 'default_predictor.pth')):
            print("Error: Model weights not found. Please run 'python main.py --train' first.")
            return
        if not os.path.exists(os.path.join('models', 'scaler.pkl')):
            print("Error: Scaler not found. Please run 'python main.py --train' first.")
            return

        print("Enter applicant details:")
        print("-" * 30)
        
        try:
            income = float(input("Total Income: "))
            credit = float(input("Total Loan Amount Requested: "))
            annuity = float(input("Yearly Loan Payment: "))
            days_birth = float(input("Age in Days (Negative number): "))
            days_employed = float(input("Days Employed (Negative number): "))
            ext_2 = float(input("External Credit Score 2 (0.0 to 1.0): "))
            ext_3 = float(input("External Credit Score 3 (0.0 to 1.0): "))
            
            raw_applicant = [income, credit, annuity, days_birth, days_employed, ext_2, ext_3]
            
            print("\nCalculating risk profile...")
            prob = predict_default(raw_applicant, input_dim=7)
            
            print("-" * 30)
            print(f"Risk Assessment: {prob * 100:.2f}% chance of default.")
            
            if prob > 0.5:
                print("Action: REJECT LOAN")
            else:
                print("Action: APPROVE LOAN")
                
        except ValueError:
            print("Invalid input. Please enter numerical values only.")

    if not (args.train or args.predict):
        parser.print_help()

if __name__ == "__main__":
    main()