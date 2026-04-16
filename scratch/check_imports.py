import os
import warnings
warnings.filterwarnings('ignore')

try:
    # Data
    import numpy as np
    import pandas as pd

    # Visualization
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Preprocessing
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    # Models
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    # Metrics
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, classification_report, confusion_matrix,
        roc_curve, ConfusionMatrixDisplay
    )

    # Model saving
    import joblib

    print('✓ All libraries loaded successfully!')
except ImportError as e:
    print(f'✗ Error: {e}')
