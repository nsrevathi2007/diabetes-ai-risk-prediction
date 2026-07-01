# API Backend Report

## Architecture Summary

The API backend is built using FastAPI and exposes a lightweight API layer on top of the existing ML pipeline.

- `api/app.py`: application factory that configures FastAPI, middleware, and exception handling.
- `api/startup.py`: initializes shared dependencies once at startup, including the SHAP explainer, recommendation engine, and LLM generator.
- `api/dependencies.py`: dependency injection helpers for service objects.
- `api/routes/`: route modules for root, health, model-info, prediction, recommendation, explanation, and full-analysis.
- `api/services/`: service layer encapsulating business logic and orchestrating ML/SHAP/recommendation/LLM subsystems.
- `api/schemas/`: request and response Pydantic models used for validation and structured payloads.

The architecture preserves the existing project structure and uses the ML, SHAP, recommendation, and LLM modules already present in `src/`.

## Available Endpoints

- `GET /`
- `GET /api/v1/health`
- `GET /api/v1/model-info`
- `POST /api/v1/predict`
- `POST /api/v1/recommend`
- `POST /api/v1/explain`
- `POST /api/v1/full-analysis`

## Verification Results

- `python run_api.py`: starts successfully, Uvicorn runs at `http://127.0.0.1:8000`
- Swagger UI: available at `/docs`
- ReDoc: available at `/redoc`
- `python -m pytest tests/test_api.py -q`: 8 passed

## Notes

- Existing modules were reused without rewriting core business logic.
- Only minimal export and route/schema fixes were applied.
- /api/v1/full-analysis returns a combined payload containing prediction, recommendation, SHAP explanation, and LLM explanation.
- The LLM generator uses fallback template mode when no OpenAI API key is configured.

## Compatibility Notes

- The API uses the repo virtualenv Python interpreter and dependencies.
- Some deprecation warnings are emitted from SHAP and `datetime.utcnow()` but do not affect functionality.
