# Customer Churn Prediction - MLOps Pipeline

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![MLOps](https://img.shields.io/badge/MLOps-Pipeline-orange.svg)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-green.svg)
![Docker](https://img.shields.io/badge/Docker-Container-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A complete MLOps pipeline for predicting customer churn with automated CI/CD, containerization, and deployment.

Project Overview

This project implements an end-to-end machine learning solution for predicting customer churn, following MLOps best practices with automated testing, Docker containerization, and continuous deployment.

### Key Features
- **Automated ML Pipeline**: From data preprocessing to model training
- **CI/CD Integration**: GitHub Actions for automated testing and deployment
- **Docker Containerization**: Ready-to-deploy API service
- **REST API**: FastAPI-based prediction endpoint
- **Monitoring**: Built-in health checks and metrics

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker 20.10+
- Git

### Installation

#### Option 1: Local Setup
```bash
# Clone repository
git clone https://github.com/SahilSinghG/Customer-churn-mlops.git
cd Customer-churn-mlops

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements_api.txt
Option 2: Docker
bash
# Build Docker image
docker build -f docker/Dockerfile -t churn-api:latest .

# Run container
docker run -d -p 8000:8000 --name churn-api churn-api:latest
Usage
Start the API Server
bash
# Local
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Or using Docker
docker run -d -p 8000:8000 churn-api:latest
API Endpoints
GET /health - Health check

GET /docs - Interactive API documentation (Swagger UI)

POST /predict - Make churn predictions

Example Prediction Request
bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "customer_age": 45,
      "gender": "Male",
      "tenure": 12,
      "monthly_charges": 79.99,
      "total_charges": 959.88
    }
  }'

Project Structure
text
customer-churn-mlops/
├── src/                    # Source code
│   ├── api/               # FastAPI application
│   ├── models/            # ML models and training
│   ├── preprocessing/     # Data preprocessing
│   └── utils/             # Utility functions
├── tests/                 # Test suite
├── docker/                # Docker configuration
│   └── Dockerfile
├── data/                  # Dataset (add .gitignore)
├── notebooks/             # Jupyter notebooks for exploration
├── .github/workflows/     # CI/CD pipelines
│   ├── ci.yml            # Continuous Integration
│   └── cd.yml            # Continuous Deployment
├── requirements_api.txt   # API dependencies
├── requirements_train.txt # Training dependencies
└── README.md             # This file

ML Pipeline

Data Collection - Customer behavior and demographic data

Preprocessing - Feature engineering and normalization

Model Training - XGBoost/LightGBM for classification

Evaluation - Accuracy, Precision, Recall, ROC-AUC

Serving - REST API for real-time predictions

CI/CD Pipeline
Continuous Integration (CI)
Trigger: Push to any branch, Pull Requests

Actions:

Code linting (Black, Flake8)

Unit testing (pytest)

Coverage reporting

Docker build test

Continuous Deployment (CD)
Trigger: Push to main branch, version tags

Actions:

Docker image build

Push to Docker Hub

Deployment to cloud (AWS/Azure/GCP)

Docker Deployment
Build Image
bash
docker build -f docker/Dockerfile -t churn-api:latest .
Run Container
bash
docker run -d \
  -p 8000:8000 \
  --name churn-api \
  -e ENVIRONMENT=production \
  churn-api:latest
Docker Compose (for development)
yaml
version: '3.8'
services:
  churn-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=development
    volumes:
      - ./src:/app/src

Model Performance
Metric	Score
Accuracy	0.89
Precision	0.87
Recall	0.85
F1-Score	0.86
ROC-AUC	0.92

Testing
Run the test suite:

bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov-report=html

Monitoring & Logging
Health Endpoint: GET /health

Metrics: Prometheus metrics endpoint

Logging: Structured JSON logging

Alerting: Integration with monitoring tools

Development
Setup Development Environment
bash
# Install development dependencies
pip install -r requirements_api.txt
pip install black flake8 pytest

# Pre-commit hooks (optional)
pre-commit install
Code Style
bash
# Format code
black src/

# Lint code
flake8 src/

# Type checking (if using type hints)
mypy src/

Contributing
Fork the repository

Create a feature branch (git checkout -b feature/AmazingFeature)

Commit changes (git commit -m 'Add AmazingFeature')

Push to branch (git push origin feature/AmazingFeature)

Open a Pull Request

🙏 Acknowledgments
Dataset: IBM Telco Customer Churn

Inspired by MLOps best practices

Built with FastAPI, Scikit-learn, and Docker

Contact

Sahil Guleria - [LINKEDIN](https://www.linkedin.com/in/sahil-guleria-4b22511bb/)

Project Link: https://github.com/SahilSinghG/Customer-churn-mlops