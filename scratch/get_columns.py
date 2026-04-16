import joblib
import os

PREPROCESSOR_PATH = "models/preprocessor.pkl"

if os.path.exists(PREPROCESSOR_PATH):
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    try:
        # Logistic Regression best_model.pkl (small) vs xgboost (large)
        print(f"Preprocessor object: {preprocessor}")
        
        if hasattr(preprocessor, 'transformers_'):
            cols = []
            for name, trans, columns in preprocessor.transformers_:
                if name != 'remainder':
                    print(f"Transformer {name} uses columns: {columns}")
                    cols.extend(columns)
            print("Full expected column order for numpy array input:", cols)
        
        # Check if we can transform a simple list/array
        import numpy as np
        # Create a dummy row matching the features
        dummy = np.zeros((1, 19)) # There are 19 features in Telco dataset
        # X = preprocessor.transform(dummy) # This might fail if it needs a DF
        
    except Exception as e:
        print(f"Error: {e}")
