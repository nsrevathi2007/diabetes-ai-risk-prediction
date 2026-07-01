# AGENTS.md

## Pipeline execution order (sequential, each step depends on prior outputs)

```
run_data_pipeline.py → run_preprocessing.py → run_eda.py
→ run_training.py → run_optimization.py → run_explainability.py
→ run_recommendations.py → run_llm.py
```

All runners are invoked from project root: `python run_*.py`

## Run from project root

No `setup.py`/`pyproject.toml`. Always run commands from `D:\Projects\diabetes-ai`. `run_data_pipeline.py` does `sys.path.insert(0, ...)`; the other runners rely on the working directory.

## Tests

`pytest tests/` or `pytest tests/test_<module>.py`. Each test file creates its own `sample_dataframe()` fixture inline. Tests use `tempfile.TemporaryDirectory` for file I/O.

```
pytest tests/                           # all tests
pytest tests/test_training.py -x -v     # single file, verbose, stop on first fail
```

## Stub/skeleton modules (NotImplementedError)

- `src/inference/predictor.py` — `DiabetesRiskPredictor.predict()`
- `src/monitoring/tracker.py` — `MonitoringTracker.record()`
- `src/utils/config.py` — `load_config()`
- `src/utils/environment.py` — `load_environment()`
- `api/v1/routes.py` — `POST /api/v1/predict`

The API `/health` endpoint works; everything else requires implementation.

## Architecture

| Directory | State | Purpose |
|---|---|---|
| `src/data_ingestion/` | Implemented | CSV loader, validator, schema, quality reports |
| `src/preprocessing/` | Implemented | Cleaner, feature classifier/engineering/selector/scaler, imbalance handler |
| `src/eda/` | Implemented | Overview, target analysis, correlations, healthcare viz, ranking |
| `src/training/` | Implemented | Trainer, model factory, evaluator, metrics, cross-val, comparison |
| `src/optimization/` | Implemented | Optuna, threshold/imbalance optimization, MLflow tracking |
| `src/explainability/` | Implemented | SHAP explainer, global/local explanations, visualizer |
| `src/recommendations/` | Implemented | Engine, profile builder, rules, templates, payload schema |
| `src/llm/` | Implemented | OpenAI provider + template fallback, safety layer |
| `src/inference/` | **Stub** | Prediction serving not yet implemented |
| `src/monitoring/` | **Stub** | Drift/performance tracking not yet implemented |
| `api/` | Partial | FastAPI app with v1 routes, schemas, middleware, error handlers |
| `frontend/` | Empty | Reserved for dashboard UI |
| `deployment/` | Empty | Reserved for Docker/K8s manifests |

## Config

- `configs/config.yaml` — data paths, schema (22 columns from BRFSS 2015), validation settings
- `configs/recommendation_config.yaml` — risk thresholds (low: 0.25, moderate: 0.50, high: 0.75), SHAP value filters
- `.env` (copy from `.env.example`) — API_HOST, API_PORT, MLFLOW_TRACKING_URI, OPENAI_API_KEY

## Key conventions

- **Model serialization**: joblib (`.joblib` files in `artifacts/models/`)
- **Dataset**: `data/raw/diabetes_binary_health_indicators_BRFSS2015.csv` → cleaned → `data/processed/diabetes_binary_health_indicators_cleaned.csv`
- **Outputs**: `artifacts/` (models, results, explanations, recommendations), `reports/` (statistics, figures, SHAP, model_reports)
- **Logging**: via `src.utils.logging.configure_logger()` — file + console handler, `logs/` directory
- **`src/__init__.py`** uses `try/except ImportError` guards — modules that fail to import are silently skipped
- **LLM module**: uses `OPENAI_API_KEY` env var; falls back to template-based generation when no key is set
- **SHAP explainer** loads pre-saved model artifacts (does not re-train)
- **Test patterns**: pytest, no conftest.py, each file self-contained with inline fixtures
