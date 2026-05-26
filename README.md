# ZepIris

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.127%2B-green)](https://fastapi.tiangolo.com/)
[![Poetry](https://img.shields.io/badge/Poetry-2.x-blueviolet)](https://python-poetry.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 👁️ ZepIris — Face Authentication at Scale

**No OTPs. No registers. No buddy punching. Just a selfie.**

ZepIris is Zepto's purpose-built face authentication platform — open-sourced for teams running identity verification at operational scale.

It handles the full pipeline: face detection, embedding generation, vector search, and spoof/blur/nudity flagging. Designed to work on budget smartphones, in low light, under high concurrency.

If you're running attendance or identity workflows at scale and don't want to stitch together multiple vendors, this is it.

**Current version:** v1.0.0. URL paths use `/v1/...` for the HTTP API; OpenAPI `info.version` and the Python package follow semver (`pyproject.toml` / `zepiris.version`).

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [Database Schema](#database-schema)
- [Configuration](#configuration)
- [Development](#development)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Documentation](#documentation)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [Support & Community](#support--community)

---

## Overview

ZepIris simplifies face recognition and verification workflows by providing:

- **Pre-integrated face embeddings** using InsightFace's buffalo_l model (512-dimensional vectors)
- **1-to-N face vector search** via Milvus for fast COSINE similarity matching
- **Automated content safety checks**: nudity detection, anti-spoofing, blur detection — all via a dedicated ML inference service
- **Multi-tenant support** — per-tenant face enrollment and search with isolated namespaces
- **Full CRUD operations** — insert, upsert, delete, and get for face records
- **Production-ready microservice architecture** with independent scaling for ML inference
- **S3-compatible image storage** via MinIO
- **REST API** with OpenAPI/Swagger auto-documentation and `requestId` traceability

### Use Cases

- **Attendance Tracking** — Enroll employee faces, query against live camera feeds
- **Onboarding Workflows** — Identity verification with liveness detection
- **Face-Based Access Control** — 1-to-N face matching with content safety validation
- **Quality Assurance** — Automatic detection of low-quality, spoofed, or unsafe images

---

## Features


| Capability                    | Description                                                                              |
| ----------------------------- | ---------------------------------------------------------------------------------------- |
| **Face Embedding**            | Extract 512-dimensional L2-normalized embeddings using InsightFace buffalo_l             |
| **1-to-N Face Search**        | Query vectors against Milvus for fast COSINE similarity matching                         |
| **Full CRUD API**             | Insert, upsert, delete, and retrieve face records with multi-tenant isolation            |
| **Content Safety (ML)**       | Nudity, spoof/deepfake, and blur detection via dedicated ML inference microservice       |
| **Multi-Tenant**              | Per-tenant face enrollment and search with isolated namespaces                           |
| **Microservice Architecture** | Separate ML inference service (port 8001) scales independently from main API (port 8000) |
| **Image Storage**             | S3-compatible MinIO integration for persistent image archival                            |
| **REST API**                  | FastAPI with OpenAPI/Swagger docs, `requestId` on every response                         |
| **Docker Ready**              | Multi-stage Dockerfiles for both services + Docker Compose with all dependencies         |
| **Configurable Thresholds**   | Fine-tune quality checks (blur sensitivity, spoof threshold, nudity confidence)          |


---

## Architecture

ZepIris consists of **two independent FastAPI microservices** that communicate via HTTP:

```
┌─────────────────────────────────────────────────┐
│  Client / Application                           │
└──────────────┬──────────────────────────────────┘
               │ (REST API)
      ┌────────▼──────────────────────────────┐
      │  Main API (port 8000)                 │
      │  ├─ POST /v1/faces/search             │
      │  ├─ POST /v1/faces/insert             │
      │  ├─ POST /v1/faces/upsert             │
      │  ├─ DELETE /v1/faces/delete            │
      │  ├─ GET  /v1/faces/get/{face_id}      │
      │  ├─ GET  /healthz                     │
      │  └─ GET  /readyz                      │
      └───┬────────────────┬──────────────────┘
          │                │
       ┌──▼──┐          ┌──▼─────┐
       │MinIO│          │ Milvus │
       │ S3  │          │ Vector │
       │Store│          │ Store  │
       └─────┘          └────────┘
                           │
      ┌────────────────────▼──────────────────────────┐
      │ ML Inference (port 8001)                      │
      │ ├─ POST /v1/face/embed      (Face Embedding)  │
      │ ├─ POST /v1/iqa/nsfw_check  (NSFW Detection)  │
      │ ├─ POST /v1/iqa/spoof_check (Spoof Detection) │
      │ ├─ POST /v1/iqa/blur_check  (Blur Detection)  │
      │ └─ POST /v1/iqa/assess      (Combined IQA)    │
      └───────────────────────────────────────────────┘
```

### Main API Service (Port 8000)

**Responsibilities:**

- Handle user-facing CRUD and search endpoints under `/v1/faces/`
- Manage image uploads and storage via MinIO
- Coordinate with ML inference service for IQA and embedding extraction
- Index face embeddings in Milvus with multi-tenant support
- Return structured responses with `requestId` for traceability

**Dependencies:**

- FastAPI, Uvicorn, Pydantic
- Milvus vector database (with Etcd for metadata)
- MinIO (S3-compatible object storage)
- HTTPx (for ML service communication)

### ML Inference Service (Port 8001)

**Responsibilities:**

- Run independent, parallelizable ML workloads
- Maintain 4 PyTorch models in memory:
  - **Face Embedding** — InsightFace buffalo_l (640×640 input → 512-d output)
  - **Nudity Detection** — MobileNetV2 (2-class classifier)
  - **Spoof Detection** — MobileNetV3-Large (liveness detection)
  - **Blur Detection** — ResNet18 (image quality assessment)
- Expose HTTP endpoints for individual or combined inference
- Combined IQA runs all 3 quality checks in parallel via `ThreadPoolExecutor`

**Benefits:**

- Scale independently — run on GPU hardware if needed
- Reuse models across requests — no repeated loading
- Parallel execution — run all 3 quality checks simultaneously

---

## Quick Start

### Prerequisites

- **Python 3.10–3.14** (tested on 3.10–3.14)
- **Poetry 2.x** for dependency management ([install here](https://python-poetry.org/docs/#installation))
- **Docker & Docker Compose** (v1.29+; recommended for all-in-one setup)
- **4GB+ RAM** for Milvus, **10GB+ free disk space**

Check your versions:

```bash
python3 --version
poetry --version
```

### Docker Compose (One Command)

For a fully containerized local setup:

```bash
# Clone and navigate
git clone <repository-url>
cd zepiris

# Start all services (Milvus, MinIO, Etcd, API, ML inference)
docker-compose up -d

# Verify health
curl http://localhost:8000/healthz

# Open API documentation
# Visit: http://localhost:8000/docs

# Stop services
docker-compose down
```

The Compose file sets `name: zepiris`, so images are tagged `zepiris-api` and `zepiris-ml-inference` regardless of clone directory name.

This starts:

- Main API (port 8000)
- ML inference (port 8001)
- Milvus (port 19530)
- MinIO (host port 9002 → container 9000, console on 9001)
- Etcd (metadata store for Milvus)

---

## API Endpoints

### Interactive Documentation

Once running, visit these URLs in your browser:

- **Main API**: [http://localhost:8000/docs](http://localhost:8000/docs) (Swagger UI)
- **ML Inference**: [http://localhost:8001/docs](http://localhost:8001/docs) (Swagger UI)

### Main API (`/v1/faces/`)

#### Health & Readiness

```bash
curl http://localhost:8000/healthz
# {"status": "ok"}

curl http://localhost:8000/readyz
# {"status": "ok"}
```

#### Insert a Face

Register a new face with an ID and tenant:

```bash
curl -X POST http://localhost:8000/v1/faces/insert \
  -F "id=employee_001" \
  -F "tenant=acme_corp" \
  -F "file=@face.jpg"
```

**Response:**

```json
{
  "requestId": "a1b2c3d4-e5f6-...",
  "imageQualityAssessment": {
    "passed": true,
    "nsfw": {"is_safe": true, "probability": 0.98},
    "spoof": {"is_live": true, "probability": 0.95},
    "blur": {"is_sharp": true, "probability": 0.90}
  },
  "userOperationResult": {
    "operation": "INSERT",
    "status": "success"
  }
}
```

**Parameters:**

- `id` (required) — unique face identifier
- `tenant` (required) — tenant namespace for isolation
- `file` (required) — JPEG/PNG image file (max 5 MB)
- Returns `409 Conflict` if a face with the same `id` already exists
- Returns `422 Unprocessable Entity` if IQA fails or no face detected

#### Search Similar Faces

Upload an image and find matching faces in the database:

```bash
curl -X POST "http://localhost:8000/v1/faces/search?top_k=5" \
  -F "id=query_001" \
  -F "tenant=acme_corp" \
  -F "file=@query_face.jpg"
```

**Response:**

```json
{
  "requestId": "d4e5f6a7-b8c9-...",
  "imageQualityAssessment": {
    "passed": true,
    "nsfw": {"is_safe": true, "probability": 0.99},
    "spoof": {"is_live": true, "probability": 0.97},
    "blur": {"is_sharp": true, "probability": 0.92}
  },
  "searchResult": {
    "matches": [
      {"id": "employee_001", "score": 0.92}
    ]
  }
}
```

**Parameters:**

- `id` (required, form) — identifier for this query
- `tenant` (required, form) — tenant namespace
- `file` (required, form) — JPEG/PNG image file
- `top_k` (optional, query, default: 5) — number of matches to return
- `threshold` (optional, query) — minimum similarity score; defaults to `ZEPIRIS_MILVUS_SEARCH_THRESHOLD`
- Returns `200 OK` with empty `matches` array if IQA fails or no face detected

#### Upsert a Face

Insert or update a face record:

```bash
curl -X POST http://localhost:8000/v1/faces/upsert \
  -F "id=employee_001" \
  -F "tenant=acme_corp" \
  -F "file=@updated_face.jpg"
```

**Response:** Same structure as Insert, with `"operation": "UPSERT"`.

#### Delete a Face

Remove a face record by ID:

```bash
curl -X DELETE "http://localhost:8000/v1/faces/delete?id=employee_001"
```

**Response:**

```json
{
  "requestId": "f6a7b8c9-d0e1-...",
  "userOperationResult": {
    "operation": "DELETE",
    "status": "success"
  }
}
```

#### Get Face Metadata

Retrieve a face record by ID:

```bash
curl http://localhost:8000/v1/faces/get/employee_001
```

**Response:**

```json
{
  "face_id": "employee_001",
  "tenant": "acme_corp",
  "object_key": "faces/employee_001"
}
```

### ML Inference API (`/v1/`)

All POST endpoints accept JSON bodies with base64-encoded images.

#### Base Payload Format

```json
{
  "image_b64": "<base64-encoded image bytes>"
}
```

**Example:** Encode an image to base64:

```bash
base64 -i face.jpg | pbcopy   # macOS
cat face.jpg | base64         # Linux
```

#### Health Check

```bash
curl http://localhost:8001/healthz
# {"status": "ok"}
```

#### NSFW Detection

Check if image contains nudity/NSFW content:

```bash
curl -X POST http://localhost:8001/v1/iqa/nsfw_check \
  -H "Content-Type: application/json" \
  -d '{"image_b64": "..."}'
```

**Response:**

```json
{
  "is_safe": true,
  "probability": 0.98
}
```

#### Spoof Detection

Check if face is real or spoofed/deepfake:

```bash
curl -X POST http://localhost:8001/v1/iqa/spoof_check \
  -H "Content-Type: application/json" \
  -d '{"image_b64": "..."}'
```

**Response:**

```json
{
  "is_live": true,
  "probability": 0.95
}
```

#### Blur Detection

Check if face image is sharp enough:

```bash
curl -X POST http://localhost:8001/v1/iqa/blur_check \
  -H "Content-Type: application/json" \
  -d '{"image_b64": "..."}'
```

**Response:**

```json
{
  "is_sharp": true,
  "probability": 0.90
}
```

#### Face Embedding

Generate a 512-dimensional face embedding:

```bash
curl -X POST http://localhost:8001/v1/face/embed \
  -H "Content-Type: application/json" \
  -d '{"image_b64": "..."}'
```

**Response:**

```json
{
  "face_detected": true,
  "embedding": [0.123, -0.456, 0.789, "..."],
  "embedding_dim": 512
}
```

#### Combined Assessment (IQA)

Run all 3 quality checks in parallel:

```bash
curl -X POST http://localhost:8001/v1/iqa/assess \
  -H "Content-Type: application/json" \
  -d '{"image_b64": "..."}'
```

**Response:**

```json
{
  "passed": true,
  "nsfw": {"is_safe": true, "probability": 0.98},
  "spoof": {"is_live": true, "probability": 0.95},
  "blur": {"is_sharp": true, "probability": 0.90}
}
```

`passed` is `true` when: `nsfw.is_safe AND spoof.is_live AND blur.is_sharp`.

---

## Database Schema

**Collection:** `zepiris_faces` (Milvus)


| Field        | Type              | Purpose               |
| ------------ | ----------------- | --------------------- |
| `face_id`    | VARCHAR(128)      | Primary key           |
| `tenant`     | VARCHAR(256)      | Multi-tenancy support |
| `object_key` | VARCHAR(512)      | MinIO image path      |
| `embedding`  | FLOAT_VECTOR(512) | Face embedding vector |


**Index:** FLAT with COSINE similarity metric.

---

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and customize. All settings use environment variable prefixes.

#### Main API Service (`ZEPIRIS_`*)


| Variable | Default | Description |
|----------|---------|-------------|
| `ZEPIRIS_API_TITLE` | `ZepIris` | API title (shown in docs) |
| `ZEPIRIS_API_VERSION` | `1.0.0` | API version (OpenAPI `info.version`) |
| `ZEPIRIS_API_HOST` | `0.0.0.0` | Bind host |
| `ZEPIRIS_API_PORT` | `8000` | Bind port |
| `ZEPIRIS_MINIO_ENDPOINT` | `localhost:9002` | MinIO S3 host:port |
| `ZEPIRIS_MINIO_ACCESS_KEY` | `minioadmin` | MinIO access key |
| `ZEPIRIS_MINIO_SECRET_KEY` | `minioadmin` | MinIO secret key |
| `ZEPIRIS_MINIO_BUCKET` | `zepiris` | S3 bucket name |
| `ZEPIRIS_MINIO_SECURE` | `false` | Use TLS for MinIO |
| `ZEPIRIS_MILVUS_HOST` | `localhost` | Milvus vector database host |
| `ZEPIRIS_MILVUS_PORT` | `19530` | Milvus port |
| `ZEPIRIS_MILVUS_COLLECTION` | `zepiris_faces` | Milvus collection name |
| `ZEPIRIS_MILVUS_EMBEDDING_DIM` | `512` | Face embedding dimension |
| `ZEPIRIS_MILVUS_SEARCH_THRESHOLD` | `0.5` | Default COSINE similarity threshold for search |
| `ZEPIRIS_ML_INFERENCE_SERVICE_URL` | *(required)* | URL of the ML inference service (e.g. `http://localhost:8001`) |

> **Note:** `ZEPIRIS_ML_INFERENCE_SERVICE_URL` is **required**. The main API will not start without it. Set it to `http://ml-inference:8001` in Docker Compose or `http://localhost:8001` when running locally.

#### ML Inference Service (`ML_SERVICE_`*)


| Variable                             | Default                        | Description                                  |
| ------------------------------------ | ------------------------------ | -------------------------------------------- |
| `ML_SERVICE_HOST`                    | `0.0.0.0`                      | Bind host                                    |
| `ML_SERVICE_PORT`                    | `8001`                         | Bind port                                    |
| `ML_SERVICE_ML_DEVICE`               | `cpu`                          | Inference device: `cpu`, `cuda:0`, `mps`     |
| `ML_SERVICE_NSFW_LOCAL_MODEL_PATH`   | `/app/models/nsfw_model.pth`   | NSFW model file                              |
| `ML_SERVICE_NSFW_HF_REPO_ID`         | ``                             | HuggingFace repo for NSFW model (optional)   |
| `ML_SERVICE_NSFW_THRESHOLD`          | `0.5`                          | NSFW classification threshold (0–1)          |
| `ML_SERVICE_SPOOF_LOCAL_MODEL_PATH`  | `/app/models/spoof_model.pth`  | Spoof model file                             |
| `ML_SERVICE_SPOOF_HF_REPO_ID`        | ``                             | HuggingFace repo for spoof model (optional)  |
| `ML_SERVICE_SPOOF_THRESHOLD`         | `0.5`                          | Spoof classification threshold (0–1)         |
| `ML_SERVICE_BLUR_LOCAL_MODEL_PATH`   | `/app/models/blur_model.pth`   | Blur model file                              |
| `ML_SERVICE_BLUR_HF_REPO_ID`         | ``                             | HuggingFace repo for blur model (optional)   |
| `ML_SERVICE_BLUR_THRESHOLD`          | `0.5`                          | Blur classification threshold (0–1)          |
| `ML_SERVICE_FACE_EMBEDDING_DIM`      | `512`                          | Face embedding dimension                     |
| `ML_SERVICE_FACE_DETECTION_WIDTH`    | `640`                          | Face detection input width                   |
| `ML_SERVICE_FACE_DETECTION_HEIGHT`   | `640`                          | Face detection input height                  |
| `ML_SERVICE_FACE_AREA_THRESHOLD`     | `0.01`                         | Minimum face area (fraction of image)        |


### Example: GPU Inference

To enable GPU inference, set:

```bash
export ML_SERVICE_ML_DEVICE=cuda:0
poetry run zepiris-ml-inference-api
```

Or in `.env`:

```bash
ML_SERVICE_ML_DEVICE=cuda:0
```

Ensure PyTorch CUDA version matches your GPU driver.

---

## Development

### Project Structure

```
zepiris/
├── pyproject.toml              # Project metadata, dependencies
├── poetry.lock                 # Locked dependency versions
├── poetry.toml                 # Poetry config (in-project .venv)
├── .env.example                # Environment variable template
├── Dockerfile                  # Main service container
├── ml_inference.Dockerfile     # ML service container
├── docker-compose.yml          # All services (MinIO, Etcd, Milvus, ML, API)
│
├── zepiris/
│   ├── main.py                 # Main FastAPI app factory + lifespan
│   ├── config.py               # Pydantic settings (ZEPIRIS_* prefix)
│   ├── deps.py                 # FastAPI dependency injection
│   ├── exceptions.py           # Domain exceptions (DuplicateFaceIdError, etc.)
│   ├── exception_handlers.py   # Error response formatting
│   │
│   ├── api/routes/
│   │   ├── __init__.py         # build_api_router()
│   │   ├── face.py             # /v1/faces/ search, insert, upsert, delete, get
│   │   └── health.py           # GET /healthz, /readyz
│   │
│   ├── ml_inference/
│   │   ├── app.py              # ML service FastAPI app + MLServiceSettings
│   │   ├── routes.py           # ML endpoints (/v1/iqa/nsfw_check, /spoof_check, /blur_check, /face/embed, /iqa/assess)
│   │   ├── deps.py             # ML service dependency injection (503 if model missing)
│   │   ├── base.py             # Base ModelService + ModelServiceConfig
│   │   ├── face_embedding.py   # FaceEmbeddingService (InsightFace)
│   │   ├── nsfw_detection.py   # NSFWDetectionService (MobileNetV2)
│   │   ├── spoof_detection.py  # SpoofDetectionService (MobileNetV3)
│   │   ├── blur_detection.py   # BlurDetectionService (ResNet18)
│   │   ├── image_quality_assessment.py # Combined IQA (ThreadPoolExecutor)
│   │   └── models/             # PyTorch model definitions
│   │
│   ├── services/
│   │   ├── embedding.py        # FaceEmbeddingProvider (ABC) + MLInferenceEmbeddingService
│   │   ├── iqa.py              # MLInferenceIQAService → HTTP /v1/iqa/assess
│   │   ├── milvus_store.py     # MilvusFaceStore (vector CRUD + search)
│   │   ├── minio_storage.py    # MinioStorageService (S3 storage)
│   │   └── ml_client.py        # MLInferenceClient (httpx, sync)
│   │
│   └── schemas/
│       ├── face.py             # SearchResponse, UpsertResponse, DeleteResponse
│       └── ml_inference.py     # FaceEmbeddingResult, IQA result schemas
│
└── models/                     # Pre-trained model weights
    ├── nsfw_model.pth
    ├── spoof_model.pth
    └── blur_model.pth
```

### Development Setup

```bash
# Activate virtual environment
source .venv/bin/activate
# or use Poetry shell
poetry shell
```

### Running with Auto-Reload

For local development with hot-reloading:

```bash
# Terminal 1 — ML service with reload
uvicorn zepiris.ml_inference.app:app --reload --host 0.0.0.0 --port 8001

# Terminal 2 — Main API with reload
ZEPIRIS_ML_INFERENCE_SERVICE_URL=http://localhost:8001 \
  uvicorn zepiris.main:app --reload --host 0.0.0.0 --port 8000
```

### Adding Dependencies

```bash
# Add a runtime dependency
poetry add httpx-oauth

# Add a development-only tool
poetry add --group dev black ruff pytest

# After changing dependencies, commit both files
git add pyproject.toml poetry.lock
git commit -m "chore: add new dependencies"
```

---

## Testing

ZepIris includes test scripts for both unit and integration testing.

### End-to-End ML Service Test

Tests all ML endpoints with real model inference:

```bash
python scripts/test_ml_service.py \
  --test-image path/to/image.jpg \
  --nsfw-model ./models/nsfw_model.pth \
  --spoof-model ./models/spoof_model.pth \
  --blur-model ./models/blur_model.pth
```

Or test against a running service:

```bash
python scripts/test_ml_service.py \
  --test-image path/to/image.jpg \
  --no-server --port 8001
```

### Direct Model Testing

Load and test models in-process without HTTP:

```bash
python scripts/test_models.py \
  --test-image path/to/image.jpg \
  --nsfw-model ./models/nsfw_model.pth \
  --spoof-model ./models/spoof_model.pth \
  --blur-model ./models/blur_model.pth
```

### Manual API Testing

Use `curl`, `httpie`, or Postman to test endpoints. Interactive Swagger docs available at `/docs` on both services.

---

## Troubleshooting

### Poetry Installation Issues

**Problem:** `poetry install` fails with "No file/folder found"

**Solution:** This is expected on first run. Poetry will resolve dependencies. Try again.

**Problem:** Wrong Python version

**Solution:** Point Poetry to the correct interpreter:

```bash
poetry env use /path/to/python3.10
poetry install
```

### Service Connection Issues

**Problem:** Main API can't connect to Milvus or MinIO

**Solution:** Verify external services are running:

```bash
# Check all services
docker compose ps

# Or check individually
curl http://localhost:9002/minio/health/live  # MinIO (host port 9002)
curl http://localhost:8001/healthz             # ML inference
```

**Problem:** Main API fails to start with "ML_INFERENCE_SERVICE_URL required"

**Solution:** Set the required environment variable:

```bash
export ZEPIRIS_ML_INFERENCE_SERVICE_URL=http://localhost:8001
```

**Problem:** ML service doesn't load models on startup

**Solution:** Check logs for model loading errors:

```bash
docker compose logs ml-inference  # if using Docker Compose
# or
poetry run zepiris-ml-inference-api       # if running locally
```

Ensure model files exist at configured paths. Default: `/app/models/{nsfw,spoof,blur}_model.pth`. If a model fails to load, that endpoint returns **503 Service Unavailable**.

### GPU Not Detected

**Problem:** ML service ignores `ML_SERVICE_ML_DEVICE=cuda`

**Solution:**

1. Verify PyTorch CUDA version matches your GPU driver:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

1. If unavailable, reinstall PyTorch with correct CUDA version:

```bash
poetry remove torch torchvision
poetry add torch torchvision --platform linux --python "^3.10"
```

### API Timeouts

**Problem:** Requests hang or timeout to ML service

**Solution:**

- Ensure ML inference service is running and healthy (`GET /healthz`)
- Check if models are still loading (can take 30-60s on first startup)
- Check network connectivity between services
- Monitor resource usage (disk space for model downloads, RAM for inference)

---

## Documentation

- **[LOCAL_SETUP_AND_TEST.md](docs/LOCAL_SETUP_AND_TEST.md)** — Step-by-step local testing guide (20-30 min)
- **[docs/API_REFERENCE.md](docs/API_REFERENCE.md)** — Complete API endpoint reference
- **[docs/CONFIGURATION.md](docs/CONFIGURATION.md)** — Environment variables & tuning
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — Code guidelines & contribution workflow
- **[.env.example](.env.example)** — Complete environment variable template

### Additional Resources

- [InsightFace Documentation](https://github.com/deepinsight/insightface) — Face embedding & detection
- [Milvus Vector Database](https://milvus.io/docs) — Vector storage & search
- [MinIO S3 SDK](https://min.io/docs/minio/kubernetes/upstream/) — Object storage
- [FastAPI Best Practices](https://fastapi.tiangolo.com/deployment/concepts/) — Web framework

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) file for details.

> The project is released under the [LICENSE](LICENSE) with contribution expectations described in [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

---

## Citation

If you use ZepIris in research or production, please cite:

```bibtex
@software{zepiris2026,
  title={ZepIris: Open-source face embedding and content safety microservice},
  author={Zepto Data Science Team},
  year={2026},
  url={https://github.com/zepto-labs/zepiris}
}
```

---

## Acknowledgments

ZepIris stands on the shoulders of excellent open-source projects:

- **[InsightFace](https://github.com/deepinsight/insightface)** — State-of-the-art face embedding models
- **[Milvus](https://milvus.io/)** — High-performance vector database
- **[FastAPI](https://fastapi.tiangolo.com/)** — Modern async Python web framework
- **[PyTorch](https://pytorch.org/)** — Deep learning framework
- **[MinIO](https://min.io/)** — S3-compatible object storage

---

## Support & Community

- **Issues:** [GitHub Issues](https://github.com/zepto-labs/zepiris/issues)
- **Discussions:** [GitHub Discussions](https://github.com/zepto-labs/zepiris/discussions)
- **Email:** [opensource@zepto.com](mailto:opensource@zepto.com)

---

**Last Updated:** April 2026
**Status:** v1.0.0
