import joblib
import os
import sys

# Add root to path to import web.app if needed, but not necessary here
MODEL_PATH = "models/best_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"

def inspect():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found")
        return
        
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        
        print(f"Model type: {type(model)}")
        print(f"Preprocessor type: {type(preprocessor)}")
        
        # Check if it's a pipeline
        if hasattr(preprocessor, 'get_feature_names_out'):
             print("Preprocessor supports get_feature_names_out")
             
    except Exception as e:
        print(f"Error loading models: {e}")

if __name__ == "__main__":
    inspect()
