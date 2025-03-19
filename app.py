from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
from transformers import pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ML models
chatbot = pipeline("text2text-generation", model="google/flan-t5-small")

# Initialize scaler for health metrics
scaler = StandardScaler()

class ChatMessage(BaseModel):
    message: str

class HealthMetrics(BaseModel):
    bloodPressure: str
    bloodSugar: str
    cholesterol: str
    heartRate: str
    temperature: str

@app.post("/api/chat")
async def chat(message: ChatMessage):
    try:
        # Generate response using the medical chatbot
        prompt = f"Answer this medical question professionally and concisely: {message.message}"
        response = chatbot(prompt, max_length=150, min_length=30)[0]['generated_text']
        
        # Add a medical disclaimer
        disclaimer = "\n\nNote: This is AI-generated advice. Please consult with a healthcare professional for accurate medical guidance."
        return {"response": response + disclaimer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict")
async def predict(metrics: HealthMetrics):
    try:
        # Extract and process health metrics
        try:
            systolic, diastolic = map(int, metrics.bloodPressure.split('/'))
            sugar = float(metrics.bloodSugar)
            chol = float(metrics.cholesterol)
            hr = float(metrics.heartRate)
            temp = float(metrics.temperature)
            
            # Create feature vector
            features = np.array([[systolic, diastolic, sugar, chol, hr, temp]])
            
            # Rule-based analysis with medical context
            concerns = []
            recommendations = []
            risk_level = "low"
            
            # Blood pressure analysis
            if systolic >= 180 or diastolic >= 120:
                concerns.append("hypertensive crisis")
                recommendations.append("seek immediate medical attention")
                risk_level = "high"
            elif systolic >= 140 or diastolic >= 90:
                concerns.append("high blood pressure")
                recommendations.append("reduce sodium intake, exercise regularly, and monitor blood pressure")
                risk_level = "moderate"
            
            # Blood sugar analysis
            if sugar >= 200:
                concerns.append("high blood sugar")
                recommendations.append("monitor carbohydrate intake and consult an endocrinologist")
                risk_level = max(risk_level, "moderate")
            elif sugar >= 140:
                concerns.append("elevated blood sugar")
                recommendations.append("monitor diet and exercise regularly")
            
            # Cholesterol analysis
            if chol >= 240:
                concerns.append("high cholesterol")
                recommendations.append("adopt a heart-healthy diet and consult with your doctor")
                risk_level = max(risk_level, "moderate")
            elif chol >= 200:
                concerns.append("borderline high cholesterol")
                recommendations.append("increase physical activity and reduce saturated fats")
            
            # Heart rate analysis
            if hr >= 120:
                concerns.append("elevated heart rate")
                recommendations.append("practice relaxation techniques and monitor stress levels")
                risk_level = max(risk_level, "moderate")
            
            # Temperature analysis
            if temp >= 103:
                concerns.append("high fever")
                recommendations.append("seek immediate medical attention")
                risk_level = "high"
            elif temp >= 100.4:
                concerns.append("fever")
                recommendations.append("rest, stay hydrated, and monitor temperature")
                risk_level = max(risk_level, "moderate")
            
            # Generate comprehensive analysis
            if concerns:
                analysis = {
                    "risk_level": risk_level,
                    "concerns": concerns,
                    "recommendations": recommendations,
                    "summary": f"Based on your results, we identified {', '.join(concerns)}. " \
                             f"Key recommendations: {'; '.join(recommendations)}. " \
                             "Please consult with your healthcare provider for proper medical advice."
                }
            else:
                analysis = {
                    "risk_level": "low",
                    "concerns": [],
                    "recommendations": ["maintain current healthy lifestyle"],
                    "summary": "Your health metrics appear to be within normal ranges. " \
                             "Continue maintaining your healthy lifestyle!"
                }
            
            return analysis
            
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid input format. Please ensure all values are entered correctly."
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
