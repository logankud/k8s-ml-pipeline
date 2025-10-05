from fastapi import FastAPI, HTTPException
import time
import logging
from typing import List
from .model import predict_probs
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ml-service")

app = FastAPI(title="ML Inference Service")

class Features(BaseModel):
    features: list[float]

# --- prometheus metrics ---
REQUESTS = Counter("inference_requests", "Total /predict requests")
LATENCY = Histogram("inference_latency", "Latency for /predict")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: Features):

    start = time.time()

    try:

        REQUESTS.inc()   # increment metric

        score = predict_probs(payload.features)
        return {"score":score}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    finally:
        LATENCY.observe(time.time() - start)  # capture latency


@app.get("/metrics")
def metrics():
    return generate_latest()
# , 200, {"Content-Type": CONTENT_TYPE_LATEST}
    

