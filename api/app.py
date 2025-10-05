from fastapi import FastAPI, HTTPException
import time
from typing import List
from .model import predict_probs
from pydantic import BaseModel


app = FastAPI(title="ML Inference Service")

class Features(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: Features):

    try:
        score = predict_probs(payload.features)
        return {"score":score}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    

