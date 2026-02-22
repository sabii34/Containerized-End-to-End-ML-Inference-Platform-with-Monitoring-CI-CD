# Containerized End-to-End ML Inference Platform (Docker + MLflow + Monitoring)

A production-style, Dockerized MLOps platform that supports model training, experiment tracking, model registry, API-based inference, and monitoring using Prometheus & Grafana

## Project Overview

This project demonstrates a complete Machine Learning system built using modern MLOps practices:

✅ Containerized multi-service architecture (Docker Compose)

✅ MLflow for experiment tracking & model registry

✅ PostgreSQL backend for metadata persistence

✅ FastAPI for model serving

✅ Prometheus for metrics collection

✅ Grafana for visualization

✅ Production-stage model loading

✅ Reproducible environment

The system allows training, registering, promoting, and serving ML models in a production-style workflow.

## System Architecture

### High-Level Flow:

Trainer → MLflow → Model Registry → FastAPI → Prometheus → Grafana
↓
PostgreSQL

### Components

| Service        | Role                                             |
| -------------- | ------------------------------------------------ |
| **Postgres**   | Stores MLflow metadata (runs, metrics, registry) |
| **MLflow**     | Experiment tracking + model registry             |
| **Trainer**    | Trains and registers ML model                    |
| **FastAPI**    | Serves production model via REST API             |
| **Prometheus** | Collects API metrics                             |
| **Grafana**    | Visualizes monitoring dashboards                 |
|  |

### Tech Stack

### Infrastructure

Docker

Docker Compose

WSL2

### Machine Learning

Python

scikit-learn

MLflow

### Backend

FastAPI

Uvicorn

### Database

PostgreSQL

### Monitoring

Prometheus

Grafana

## Machine Learning Pipeline

### Dataset

- Iris dataset (multi-class classification)

### Model

- Logistic Regression

- Preprocessing: StandardScaler

- Accuracy: ~93%

### Workflow

- Load dataset

- Train-test split

- Train pipeline (Scaler + LogisticRegression)

- Log parameters & metrics to MLflow

- Register model (IrisClassifier)

- Promote to Production

- API loads Production model

## Getting Started

1️⃣ Clone the Repository

git clone <your-repo-url>
cd <project-folder>

2️⃣ Build & Start Services

docker compose up -d --build

3️⃣ Train the Model

docker compose --profile manual run --rm trainer

4️⃣ Reload Model in API
http://localhost:8000/docs

#### Grafana default login:

- Username: admin

- Password: admin

🔁 Example API Request

POST /predict
{
"features": [5.1, 3.5, 1.4, 0.2]
}

### Future Improvements

- CI/CD integration (GitHub Actions)

- Auto model reload on promotion

- Kubernetes deployment

- Load testing

- Authentication & API security

## Author

### Saba Shahbaz

MLOps Enthusiast | IT Student
Focused on Production ML Systems & Cloud-Native ML
