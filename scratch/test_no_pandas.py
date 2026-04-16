import joblib
import numpy as np
import os

PREPROCESSOR_PATH = "models/preprocessor.pkl"

def test():
    if not os.path.exists(PREPROCESSOR_PATH):
        print("Error: No preprocessor")
        return
        
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    
    # Correct order from previous research
    cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 
            'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
            'PaperlessBilling', 'PaymentMethod']
    
    # Create a dummy row matching the columns
    # Mixed types: int, float, str
    dummy_row = [
        12, 70.0, 840.0, 0, # num
        'Male', 'No', 'No', 'Yes', 'No', 'DSL', 'No', 'No', 'No', 'No', 'No', 'No', 'Month-to-month', 'Yes', 'Electronic check' # cat
    ]
    
    try:
        # Try as a list of lists (2D array-like)
        X = preprocessor.transform([dummy_row])
        print("Success! Transform works without Pandas.")
        print("Output shape:", X.shape)
    except Exception as e:
        print(f"Failed to transform without Pandas: {e}")

if __name__ == "__main__":
    test()
