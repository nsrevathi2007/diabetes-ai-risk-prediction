# Frontend Report

## Architecture

The frontend is a Streamlit application located in `frontend/web/app.py`. It is intentionally API-driven and never calls the ML modules directly.

## Pages

- Home
- Prediction
- Explainability
- Recommendations
- Analytics
- About

## Components

- Sidebar navigation
- Configurable backend URL input
- Metric cards and success/error states
- Plotly charts
- Expanders and tabs-style content sections

## Backend Integration

The application consumes the existing FastAPI backend endpoints:

- `GET /`
- `GET /api/v1/health`
- `GET /api/v1/model-info`
- `POST /api/v1/predict`
- `POST /api/v1/recommend`
- `POST /api/v1/explain`
- `POST /api/v1/full-analysis`

## Testing Summary

- Streamlit app module exists
- Streamlit app imports successfully
