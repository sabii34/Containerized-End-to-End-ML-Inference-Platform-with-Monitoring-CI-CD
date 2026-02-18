import os
import mlflow

def load_model_from_registry():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    model_name = os.getenv("MODEL_NAME", "IrisClassifier")
    model_stage = os.getenv("MODEL_STAGE", "Production")

    mlflow.set_tracking_uri(tracking_uri)

    model_uri = f"models:/{model_name}/{model_stage}"
    model = mlflow.pyfunc.load_model(model_uri)
    return model, model_uri
