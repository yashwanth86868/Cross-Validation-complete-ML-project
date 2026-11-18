
# ==========================================
# FASTAPI - KERNEL SVM PREDICTION API
# ==========================================

import joblib
import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 1. Create FastAPI application

app = FastAPI(
    title="Social Network Ads Prediction API",
    description="Customer purchase prediction using Kernel SVM",
    version="1.0.0"
)

# 2. Load trained ML model

model = joblib.load("models/best_svm_model.pkl")


# 3. Define input validation

class CustomerData(BaseModel):

    age: int = Field(gt=0, le=120)

    estimated_salary: float = Field(
        ge=0,
        le=100000000
    )


# 4. Home endpoint

@app.get("/")
def home():

    return {
        "message": "Welcome to Kernel SVM Prediction API",
        "status": "Running"
    }


# 5. Health check endpoint

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model": "Kernel SVM"
    }


# 6. Prediction endpoint

@app.post("/predict")
def predict(data: CustomerData):

    try:

        input_data = np.array([[
            data.age,
            data.estimated_salary
        ]])

        prediction = model.predict(input_data)

        result = int(prediction[0])

        if result == 1:

            message = "Customer may purchase the product"

        else:

            message = "Customer may not purchase the product"

        return {
            "age": data.age,
            "estimated_salary": data.estimated_salary,
            "prediction": result,
            "message": message
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
