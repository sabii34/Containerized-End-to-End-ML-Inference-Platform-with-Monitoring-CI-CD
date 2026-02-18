# Containerized End-to-End ML Inference Platform (Docker + MLflow + Monitoring)

This project demonstrates a production-style MLOps system using Docker:

- Postgres as MLflow backend store
- MLflow Tracking + Model Registry
- Training container registers a model and promotes it to Production
- FastAPI inference service loads Production model from MLflow
- Prometheus + Grafana for monitoring

## Services

- MLflow: http://localhost:5000
- API Docs (Swagger): http://localhost:8000/docs
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (default: admin / admin)

## Run Core Stack

```bash
docker compose up --build -d
```
