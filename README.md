# Attribute-Prediction

Road traffic attribute prediction system combining machine learning models with a production-ready API microservice.

## Repository Structure

This repository contains two main components:

### Machine Learning & Data Science
Model training, data processing, and experimentation for road traffic attribute prediction.

**Location**: `data/`, `data_manipulation/`, `src/data_manipulation/`

**Documentation**: [ML_README.md](ML_README.md)

> **Note**: Managed by the ML team

### API Microservice
Production FastAPI service for serving predictions and managing model metadata.

**Location**: `src/app/`, `k8s/`, `.github/workflows/`

**Documentation**: [src/app/README.md](src/app/README.md)

**Features**:
- RESTful API for model metrics and imputation results
- Kubernetes deployment with CI/CD
- PostgreSQL database integration
- Comprehensive test suite (75% coverage)

## Quick Start

### API Microservice
```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your DATABASE_URL

# Run locally
uvicorn src.app.main:app --reload
```

See [src/app/README.md](src/app/README.md) for detailed setup instructions.

### ML & Data Science
See [ML_README.md](ML_README.md) for ML-specific documentation.

## Project Overview

```
Attribute-Prediction/
├── src/
│   ├── app/                    # FastAPI microservice
│   │   ├── routes/             # API endpoints
│   │   ├── services/           # Business logic
│   │   ├── database_tables/    # ORM models
│   │   └── schemas/            # Pydantic models
│   └── data_manipulation/      # ML data processing
├── data/                       # Training data & model checkpoints
├── tests/                      # Test suite
├── k8s/                        # Kubernetes manifests
├── .github/workflows/          # CI/CD pipelines
└── requirements.txt            # Python dependencies
```

## Testing

```bash
# Run all tests
pytest tests/mircro_service/ -v

# With coverage report
pytest tests/mircro_service/ --cov=src/app --cov-report=html
```

## Deployment

The API microservice is automatically deployed to Kubernetes on every push to `main`:
- Docker image: `ghcr.io/astep-aau/attribute-prediction:latest`
- Namespace: `cs-25-sw-5-06`
- Service: `attribute-prediction-service`

See [src/app/README.md#deployment](src/app/README.md#deployment) for details.

## License

See [LICENSE](LICENSE) file for details.
