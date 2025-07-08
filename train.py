import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib

# 🎯 Step 1: Dummy training data
data = pd.DataFrame({
    'website_visits': [2, 15, 8, 1, 30, 25, 3, 9, 12, 5],
    'email_click_rate': [0.1, 0.8, 0.5, 0.0, 0.95, 0.88, 0.2, 0.6, 0.3, 0.45],
    'job_title_level': [1, 3, 2, 0, 4, 4, 1, 2, 3, 1],
    'company_size': [20, 500, 100, 10, 1000, 800, 30, 200, 150, 60],
    'converted': [0, 1, 1, 0, 1, 1, 0, 1, 0, 0]
})

X = data.drop("converted", axis=1)
y = data["converted"]

# 🎯 Step 2: Train and save model
model = XGBClassifier(eval_metric="logloss")
model.fit(X, y)

joblib.dump(model, "lead_model.pkl")
print("✅ Model saved successfully!")
