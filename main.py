from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# ✅ Load trained model
try:
    model = joblib.load("lead_model.pkl")
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Failed to load model:", str(e))
    raise e

# 🚀 FastAPI app
app = FastAPI(title="Lead Scoring API", version="1.0")

# 📦 Request schema
class LeadInput(BaseModel):
    website_visits: int
    email_click_rate: float
    job_title_level: int
    company_size: int

# 🏠 Home endpoint
@app.get("/")
def root():
    return {"message": "🎯 Lead Scoring API is running. Visit /docs to test the endpoint."}

# 📤 Simulated CRM push function
def push_to_crm(lead_data: dict):
    file_exists = os.path.isfile("crm_log.csv")
    df = pd.DataFrame([lead_data])
    df.to_csv("crm_log.csv", mode='a', header=not file_exists, index=False)
    print("📝 Lead pushed to CRM log (csv).")

# 🔍 Prediction endpoint
@app.post("/score/")
def score_lead(lead: LeadInput):
    try:
        input_df = pd.DataFrame([{
            "website_visits": lead.website_visits,
            "email_click_rate": lead.email_click_rate,
            "job_title_level": lead.job_title_level,
            "company_size": lead.company_size
        }])

        prob = float(model.predict_proba(input_df)[0][1])
        score = round(prob, 4)

        lead_data = input_df.copy()
        lead_data["intent_score"] = score

        # ✅ Push to CRM log
        push_to_crm(lead_data.iloc[0].to_dict())

        return {"intent_score": score}

    except Exception as e:
        print("❌ Error during prediction:", str(e))
        raise HTTPException(status_code=500, detail=f"Internal Error: {str(e)}")
