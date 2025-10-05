from fastapi import FastAPI, HTTPException
import time


app = FastAPI(title="ML Inference Service")


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict():
    # TOD: implement inference logic
    return True

