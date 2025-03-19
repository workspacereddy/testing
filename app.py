from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="MediAssist AI Backend")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get Hugging Face API token from environment variable
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
if not HUGGINGFACE_API_TOKEN:
    raise ValueError("HUGGINGFACE_API_TOKEN environment variable is not set")

# Models for request/response
class ChatMessage(BaseModel):
    message: str

class HealthData(BaseModel):
    bloodPressure: str
    bloodSugar: float
    cholesterol: float
    heartRate: float
    symptoms: str

class PredictionResponse(BaseModel):
    prediction: str
    recommendations: List[str]
    risk_level: str

# Medical chatbot endpoint
@app.post("/chat")
async def chat_endpoint(chat_input: ChatMessage):
    try:
        # Using Hugging Face's API for medical dialogue
        API_URL = "https://api-inference.huggingface.co/models/microsoft/BioGPT"
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
        
        # Prepare the prompt with medical context
        prompt = f"Medical Assistant: You are asking about: {chat_input.message}\nResponse:"
        
        response = requests.post(
            API_URL,
            headers=headers,
            json={"inputs": prompt, "parameters": {"max_length": 150}}
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Error communicating with AI model")
            
        result = response.json()
        
        # Extract and clean the response
        ai_response = result[0]["generated_text"].replace(prompt, "").strip()
        
        return {"response": ai_response}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health prediction endpoint
@app.post("/predict")
async def predict_health(health_data: HealthData):
    try:
        # Parse blood pressure
        systolic, diastolic = map(int, health_data.bloodPressure.split('/'))
        
        # Basic risk assessment logic
        risk_factors = []
        recommendations = []
        
        # Blood pressure analysis
        if systolic >= 140 or diastolic >= 90:
            risk_factors.append("high blood pressure")
            recommendations.append("Consider consulting a healthcare provider about blood pressure management")
            
        # Blood sugar analysis
        if health_data.bloodSugar > 140:
            risk_factors.append("elevated blood sugar")
            recommendations.append("Monitor blood sugar levels and maintain a balanced diet")
            
        # Cholesterol analysis
        if health_data.cholesterol > 200:
            risk_factors.append("high cholesterol")
            recommendations.append("Consider lifestyle changes and consult with a healthcare provider")
            
        # Heart rate analysis
        if health_data.heartRate > 100:
            risk_factors.append("elevated heart rate")
            recommendations.append("Monitor heart rate and consider stress reduction techniques")
            
        # Determine risk level
        risk_level = "low"
        if len(risk_factors) >= 3:
            risk_level = "high"
        elif len(risk_factors) >= 1:
            risk_level = "moderate"
            
        # Generate prediction summary
        prediction = "Based on your health data, "
        if risk_factors:
            prediction += f"you show signs of {', '.join(risk_factors)}. "
        else:
            prediction += "your vital signs are within normal ranges. "
            recommendations.append("Continue maintaining your healthy lifestyle")
            
        return PredictionResponse(
            prediction=prediction,
            recommendations=recommendations,
            risk_level=risk_level
        )
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid blood pressure format. Use format: 120/80")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}
