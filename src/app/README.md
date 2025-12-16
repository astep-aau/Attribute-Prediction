# Attribute-Prediction API Microservice

A FastAPI-based microservice for managing machine learning model metrics, imputation results, and model artifacts. Built for deployment on Kubernetes with automated CI/CD pipelines.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Local Development](#local-development)
- [API Documentation](#api-documentation)
- [Database Schema](#database-schema)
- [Testing](#testing)
- [Deployment](#deployment)
- [Environment Variables](#environment-variables)

## Overview

This service provides a RESTful API for:
- Managing ML model types and metadata
- Storing and retrieving model performance metrics
- Managing imputation results for road traffic data
- Downloading trained model files

## Features

- **Async FastAPI** - High-performance async API with SQLAlchemy 2.0
- **PostgreSQL** - Production database with connection pooling
- **Comprehensive Testing** - 71 tests with 75% coverage (unit + integration)
- **Health Checks** - Kubernetes liveness/readiness probes with DB validation
- **Logging** - Structured logging for all operations
- **CI/CD** - Automated testing, building, and deployment via GitHub Actions
- **Kubernetes Deployment** - Production-ready manifests with resource limits

## Architecture

```
src/app/
├── main.py                 # FastAPI app with middleware and exception handlers
├── database.py             # Database engine and session management
├── routes/                 # API endpoint definitions
│   ├── model_type_routes.py
│   ├── metric_routes.py
│   ├── impute_results_routes.py
│   ├── download_model_routes.py
│   └── health_routes.py
├── services/               # Business logic layer
│   ├── model_type_utils.py
│   ├── metric_utils.py
│   ├── impute_result_utils.py
│   └── download_model_utils.py
├── database_tables/        # SQLAlchemy ORM models
├── schemas/                # Pydantic request/response models
└── exceptions.py           # Custom exception definitions
```

## Local Development

### Prerequisites

- Python 3.13+
- PostgreSQL 14+
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/astep-aau/Attribute-Prediction.git
   cd Attribute-Prediction
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv

   # Windows
   .\venv\Scripts\Activate

   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**

   Copy `.env.example` to `.env` and configure:
   ```bash
   cp .env.example .env
   ```

   Edit `.env`:
   ```
   DATABASE_URL=postgresql+asyncpg://username:password@localhost:5432/attribute_prediction
   DEBUG=true
   ```

5. **Run the application**
   ```bash
   uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8000
   ```

6. **Access the API**
   - API: http://localhost:8000
   - Interactive docs: http://localhost:8000/docs
   - OpenAPI schema: http://localhost:8000/openapi.json

## API Documentation

### Model Types

- `POST /model-types/create` - Create a new model type
- `GET /model-types/` - List all model types

### Model Metrics

- `POST /model-metrics/create` - Create model metrics
- `GET /model-metrics/{model_type}` - Get metrics for a model type

### Impute Results

- `POST /impute-results/create` - Create imputation result
- `GET /impute-results/` - Get imputation results with filters
- `GET /impute-results/road-ids/{model_id}` - Get road IDs for a model
- `GET /impute-results/timespan/{model_id}/{road_id}` - Get time range

### Downloads

- `GET /download_model/{model_id}` - Download model file (.pth)

### Health Checks

- `GET /health/` - Basic health check
- `GET /health/live` - Kubernetes liveness probe
- `GET /health/ready` - Kubernetes readiness probe (validates DB connection)

For detailed API documentation, run the server and visit `/docs`.

## Database Schema

### Tables

**model_types**
- `id` (UUID, PK)
- `name` (String, unique)

**model_metrics**
- `id` (UUID, PK)
- `model_type` (UUID, FK → model_types)
- `train_time_min` (Integer)
- `bias` (Float)
- `gap` (Float)
- `path_to_save` (String)
- `created_at` (Timestamp)

**hyperparams**
- `id` (UUID, PK)
- `model_id` (UUID, FK → model_metrics)
- `param_name` (String)
- `param_value` (String)

**loss**
- `id` (UUID, PK)
- `model_id` (UUID, FK → model_metrics)
- `type` (String)
- `loss_value` (Float)
- `loss_unit` (String)

**impute_results**
- `model_id` (UUID, PK, FK → model_metrics)
- `road_id` (String, PK)
- `tms` (BigInteger, PK) - Unix timestamp
- `value` (Float)
- `imputed` (Boolean)

## Testing

### Running Tests

```bash
# All tests
pytest tests/mircro_service/ -v

# With coverage
pytest tests/mircro_service/ --cov=src/app --cov-report=html

# Integration tests only
pytest tests/mircro_service/integration/ -v

# Unit tests only
pytest tests/mircro_service/unit/ -v
```

### Test Structure

- **Unit Tests** (39 tests) - Service layer logic with mocked database
- **Integration Tests** (32 tests) - Full API endpoint testing with SQLite
- **Coverage**: 75% overall, 98-100% on service utilities

See [tests/how_to_run_test.md](../../tests/how_to_run_test.md) for more details.

## Deployment

### CI/CD Pipeline

The project uses GitHub Actions for automated deployment:

#### Pull Request Workflow
`.github/workflows/pr-build-and-test.yml`
- Runs tests on every PR to `main`
- Builds Docker image
- Blocks merge if tests fail

#### Deployment Workflow
`.github/workflows/deploy.yml`
- Triggers on push to `main`
- Builds and pushes Docker image to `ghcr.io`
- Deploys to Kubernetes (cs-25-sw-5-06 namespace)
- Tags images with `:latest` and `:{git-sha}`

### Kubernetes Deployment

**Resources:**
- CPU: 300m request / 600m limit
- Memory: 512Mi request / 1Gi limit
- Replicas: 1 (Recreate strategy for zero-downtime with quota constraints)

**Manifests:**
- `k8s/deployment.yaml` - Main deployment configuration
- `k8s/service.yaml` - ClusterIP service (port 80 → 8000)
- `k8s/secret.yaml` - Database credentials

**Access:**
- Within cluster: `http://attribute-prediction-service`
- Other namespaces: `http://attribute-prediction-service.cs-25-sw-5-06`

### Manual Deployment

```bash
# Build Docker image
docker build -t ghcr.io/astep-aau/attribute-prediction:latest .

# Push to registry
docker push ghcr.io/astep-aau/attribute-prediction:latest

# Deploy to Kubernetes (requires kubeconfig)
kubectl apply -f k8s/deployment.yaml -n cs-25-sw-5-06
kubectl apply -f k8s/service.yaml -n cs-25-sw-5-06
```

## Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DATABASE_URL` | PostgreSQL connection string | - | Yes |
| `DEBUG` | Enable SQL query logging | `false` | No |
| `ENV` | Environment (local/cluster) | `local` | No |

### Example Configurations

**Local Development:**
```bash
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/dbname
DEBUG=true
```

**Cluster (Kubernetes):**
```yaml
env:
  - name: ENV
    value: "cluster"
  - name: DEBUG
    value: "false"
  - name: DATABASE_URL
    valueFrom:
      secretKeyRef:
        name: db-secret
        key: DATABASE_URL
```
