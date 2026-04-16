# CustomerPulse 🔮

> Customer Churn Prediction — Machine Learning Course Project

## Project Structure

```
CustomerPulse/
├── notebook/
│   └── CustomerPulse_Churn_Analysis.ipynb   ← All ML work (EDA, training, evaluation)
├── web/
│   ├── app.py                               ← Flask web app logic
│   ├── templates/                           ← Web UI
│   └── static/                              ← Styling
├── scratch/                                 ← Scripts used to fix Vercel deployment limits
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv ← IBM Telco dataset
├── models/                                  ← Saved models (.pkl)
├── requirements.txt                         ← Production dependencies (Slimmed for Vercel)
├── requirements-dev.txt                     ← Full dependencies (Local/Notebook)
├── app.py                                   ← Main entry point (Root)
└── README.md
```

## Dataset

**IBM Telco Customer Churn** — 7,043 customers with 21 features including demographics, account info, and services subscribed. Target: `Churn` (Yes/No).

## Setup

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Running the Notebook

```bash
jupyter notebook notebook/CustomerPulse_Churn_Analysis.ipynb
```

The notebook covers:
1. **Data Exploration** — distribution, missing values, correlation
2. **Preprocessing** — encoding, scaling, train/test split
3. **Model Training** — Logistic Regression, Random Forest, XGBoost
4. **Evaluation** — Accuracy, Precision, Recall, F1, ROC-AUC
5. **Comparison** — side-by-side metrics table + visualizations
6. **Model Export** — saves best model to `models/` folder

## Running the Web App

```bash
# Option 1: Run from root (Recommended)
python app.py

# Option 2: Traditional Flask way
cd web
python app.py
```

Then open http://localhost:5000 in your browser.

## Models Trained

| Model | Description |
|---|---|
| Logistic Regression | Linear baseline model |
| Random Forest | Ensemble of decision trees |
| XGBoost | Gradient boosting (best performer typically) |

## Deployment Notes (Vercel)

Dự án đã được tối ưu hóa để vượt qua giới hạn **500MB Lambda Storage** của Vercel:
- Thư mục `scratch/` chứa các script kiểm tra và cấu trúc lại hệ thống để giảm kích thước.
- Thư viện `xgboost` đã được gỡ bỏ trong bản production để tiết kiệm ~200MB không gian.
- Sử dụng mô hình Logistic Regression làm mặc định cho ứng dụng Web trên Cloud để đảm bảo tính ổn định.

## Author

Machine Learning Course Project
