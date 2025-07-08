from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

# 🎯 Training
data = pd.DataFrame({
    'website_visits': [2, 15, 8, 1, 30, 25, 3, 9, 12, 5],
    'email_click_rate': [0.1, 0.8, 0.5, 0.0, 0.95, 0.88, 0.2, 0.6, 0.3, 0.45],
    'job_title_level': [1, 3, 2, 0, 4, 4, 1, 2, 3, 1],
    'company_size': [20, 500, 100, 10, 1000, 800, 30, 200, 150, 60],
    'converted': [0, 1, 1, 0, 1, 1, 0, 1, 0, 0]
})
X = data.drop("converted", axis=1)
y = data["converted"]
model = XGBClassifier(eval_metric='logloss')
model.fit(X, y)
joblib.dump(model, "lead_model.pkl")

# ✅ Load model
model = joblib.load("lead_model.pkl")

# 🚀 API
app = FastAPI()

class LeadInput(BaseModel):
    website_visits: int
    email_click_rate: float
    job_title_level: int
    company_size: int

@app.post("/score/")
def score_lead(lead: LeadInput):
    try:
        features = np.array([[lead.website_visits, lead.email_click_rate, lead.job_title_level, lead.company_size]])
        score = model.predict_proba(features)[0][1]
        return {"intent_score": round(score, 4)}
    except Exception as e:
        print("❌ Internal Error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))
