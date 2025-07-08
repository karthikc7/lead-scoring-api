# AI Lead Scoring API 🚀

## Overview
This is a real-time AI-based lead scoring engine built using FastAPI and XGBoost. It predicts the intent score of leads based on behavioral and demographic data.

## Features
- FastAPI backend
- XGBoost model
- Realtime `/score/` endpoint
- Swagger UI at `/docs`
- CRM Simulation: writes lead score to `crm_log.csv`

## Setup

### Train Model
```bash
python train_model.py

uvicorn main:app --reload


Then visit http://localhost:8000/docs



---

## 🧳 Final Step

- Select all files + `lead_model.pkl` (after you run `train_model.py`)
- Right-click → **Send to → Compressed (zipped) folder**
- Name it: `lead_scoring_api_project.zip`

Now you’re ready to deploy or submit!  
Let me know if you want help testing the `.zip` before submission. ​:contentReference[oaicite:0]{index=0}​
