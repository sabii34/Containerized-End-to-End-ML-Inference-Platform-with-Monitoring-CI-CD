import time
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from model_loader import load_model_from_registry

app = FastAPI(title="Containerized ML Inference API (MLflow + Docker)")

REQUESTS = Counter("api_requests_total", "Total API requests", ["endpoint"])
LATENCY = Histogram("api_latency_seconds", "Request latency in seconds", ["endpoint"])

class PredictRequest(BaseModel):
    features: list[float]  # Iris expects 4 floats

_model = None
_model_uri = None

def _try_load():
    global _model, _model_uri
    _model, _model_uri = load_model_from_registry()

@app.on_event("startup")
def startup():
    # Try load model on startup (may fail if not trained yet)
    try:
        _try_load()
    except Exception as e:
        print(f"⚠️ Model not loaded on startup (train first): {e}")

@app.get("/health")
def health():
    REQUESTS.labels(endpoint="health").inc()
    return {"status": "ok", "model_loaded": _model is not None, "model_uri": _model_uri}

@app.post("/reload")
def reload_model():
    REQUESTS.labels(endpoint="reload").inc()
    _try_load()
    return {"reloaded": True, "model_uri": _model_uri}

@app.post("/predict")
def predict(req: PredictRequest):
    REQUESTS.labels(endpoint="predict").inc()
    start = time.time()
    try:
        if _model is None:
            raise HTTPException(status_code=503, detail="Model not loaded. Run trainer then call /reload.")

        if len(req.features) != 4:
            raise HTTPException(status_code=400, detail="Iris model expects exactly 4 features.")

        x = np.array(req.features, dtype=float).reshape(1, -1)
        y = _model.predict(x)
        return {"prediction": int(y[0]), "model_uri": _model_uri}
    finally:
        LATENCY.labels(endpoint="predict").observe(time.time() - start)

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
