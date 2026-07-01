"""Streamlit healthcare dashboard for the diabetes risk prediction API."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import requests


st.set_page_config(page_title="Diabetes Care AI", page_icon="🩺", layout="wide")


DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
FEATURE_FIELDS = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "BMI",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Sex",
    "Age",
    "Education",
    "Income",
]

AGE_GROUPS = [
    ("18-24", 1),
    ("25-29", 2),
    ("30-34", 3),
    ("35-39", 4),
    ("40-44", 5),
    ("45-49", 6),
    ("50-54", 7),
    ("55-59", 8),
    ("60-64", 9),
    ("65-69", 10),
    ("70-74", 11),
    ("75-79", 12),
    ("80+", 13),
]

SEX_OPTIONS = [("Female", 0), ("Male", 1)]
GENERAL_HEALTH_OPTIONS = [("Excellent", 1), ("Very Good", 2), ("Good", 3), ("Fair", 4), ("Poor", 5)]
EDUCATION_OPTIONS = [("Never attended school", 1), ("Elementary", 2), ("Some high school", 3), ("High school graduate", 4), ("Some college", 5), ("College graduate", 6)]
INCOME_OPTIONS = [("< $10k", 1), ("$10k-$15k", 2), ("$15k-$20k", 3), ("$20k-$25k", 4), ("$25k-$35k", 5), ("$35k-$50k", 6), ("$50k-$75k", 7), ("> $75k", 8)]


@st.cache_data(show_spinner=False)
def fetch_json(url: str, timeout: int = 20) -> Any:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


@st.cache_data(show_spinner=False)
def post_json(url: str, payload: dict[str, Any], timeout: int = 30) -> Any:
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def render_sidebar() -> str:
    st.sidebar.image("https://images.unsplash.com/photo-1576091160550-2173dba999ef?auto=format&fit=crop&w=400&q=80", use_container_width=True)
    st.sidebar.title("Diabetes Care AI")
    st.sidebar.caption("Healthcare intelligence • API-driven insights")

    if "backend_url" not in st.session_state:
        st.session_state["backend_url"] = DEFAULT_BACKEND_URL

    backend_url = st.sidebar.text_input(
        "Backend URL",
        key="backend_url",
    )
    if backend_url:
        backend_url = backend_url.rstrip("/")

    page = st.sidebar.radio(
        "Navigate",
        ["Home", "Prediction", "Explainability", "Recommendations", "Analytics", "About"],
        index=0,
    )
    return page


def render_home() -> None:
    st.title("🩺 Diabetes Care AI")
    st.subheader("A modern clinical decision support dashboard")
    st.write("This interface consumes the existing FastAPI backend for risk prediction, explainability, and recommendation workflows.")

    backend_url = st.session_state.backend_url
    status = "🟢 Connected" if ping_backend(backend_url) else "🔴 Offline"
    st.metric("Backend status", status)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("API", "FastAPI")
    with col2:
        st.metric("UI", "Streamlit")
    with col3:
        st.metric("Visualization", "Plotly")

    st.success("The frontend is fully API-driven and never calls ML modules directly.")

    with st.expander("Pipeline architecture", expanded=True):
        st.markdown(
            """
            - Client → Streamlit dashboard
            - FastAPI backend → prediction, recommendation, explanation, analytics
            - Existing ML pipeline → model inference and explainability
            - Structured JSON response → UI rendering
            """
        )

    with st.expander("Available endpoints"):
        st.code("/\n/api/v1/health\n/api/v1/model-info\n/api/v1/predict\n/api/v1/recommend\n/api/v1/explain\n/api/v1/full-analysis")


def ping_backend(url: str) -> bool:
    try:
        response = requests.get(f"{url}/api/v1/health", timeout=8)
        return response.status_code == 200
    except requests.RequestException:
        return False


def build_patient_payload() -> dict[str, Any]:
    values: dict[str, Any] = {}
    for field in FEATURE_FIELDS:
        value = st.session_state.get(field)
        if value is None:
            value = 0
        if field == "BMI":
            values[field] = float(value)
        else:
            values[field] = int(value)
    return values


def get_default_value(field: str) -> Any:
    defaults = {
        "HighBP": 0,
        "HighChol": 0,
        "CholCheck": 1,
        "BMI": 27.5,
        "Smoker": 0,
        "Stroke": 0,
        "HeartDiseaseorAttack": 0,
        "PhysActivity": 1,
        "Fruits": 1,
        "Veggies": 1,
        "HvyAlcoholConsump": 0,
        "AnyHealthcare": 1,
        "NoDocbcCost": 0,
        "GenHlth": 3,
        "MentHlth": 0,
        "PhysHlth": 0,
        "DiffWalk": 0,
        "Sex": 0,
        "Age": 7,
        "Education": 4,
        "Income": 5,
    }
    return defaults.get(field, 0)


def render_binary_choice(label: str, field: str, help_text: str) -> None:
    current_value = int(st.session_state.get(field, 0))
    choice = st.radio(
        label,
        ["No", "Yes"],
        index=current_value,
        horizontal=True,
        key=f"{field}_radio",
        help=help_text,
    )
    st.session_state[field] = 1 if choice == "Yes" else 0


def render_choice_dropdown(label: str, field: str, options: list[tuple[str, int]], help_text: str) -> None:
    current_value = int(st.session_state.get(field, options[0][1]))
    labels = [option[0] for option in options]
    values = [option[1] for option in options]
    current_index = values.index(current_value) if current_value in values else 0
    selected_label = st.selectbox(label, labels, index=current_index, key=f"{field}_select", help=help_text)
    st.session_state[field] = next(value for label, value in options if label == selected_label)


def render_prediction() -> None:
    st.title("Prediction workspace")
    st.write("Complete the intake form to receive a real-time risk assessment and a tailored care summary.")
    st.caption("These clinical choices are translated into the model's expected format automatically.")

    for field in FEATURE_FIELDS:
        if field not in st.session_state:
            st.session_state[field] = get_default_value(field)

    if st.button("Reset form", type="secondary"):
        for field in FEATURE_FIELDS:
            st.session_state.pop(field, None)
        st.rerun()

    with st.form("patient_form"):
        with st.expander("Basic Information", expanded=True):
            col_a, col_b, col_c = st.columns([1.2, 1.2, 1.4])
            with col_a:
                render_choice_dropdown("Age group", "Age", AGE_GROUPS, "Select the age bracket that best matches the patient profile.")
            with col_b:
                render_choice_dropdown("Sex", "Sex", SEX_OPTIONS, "This field aligns with the model's binary sex encoding.")
            with col_c:
                bmi_value = st.slider(
                    "BMI (kg/m²)",
                    min_value=12.0,
                    max_value=80.0,
                    value=float(st.session_state.get("BMI", 27.5)),
                    step=0.1,
                    format="%.1f",
                    help="Body mass index is a common screening measure used in wellness and diabetes risk models.",
                )
                st.session_state["BMI"] = float(bmi_value)

        with st.expander("Medical History", expanded=True):
            col_a, col_b = st.columns(2)
            with col_a:
                render_binary_choice("High blood pressure", "HighBP", "Indicates whether the patient has been diagnosed with high blood pressure.")
                render_binary_choice("High cholesterol", "HighChol", "Indicates whether cholesterol levels are elevated.")
                render_binary_choice("History of stroke", "Stroke", "Recent or past stroke history increases cardiovascular and diabetes-related risk.")
            with col_b:
                render_binary_choice("Heart disease", "HeartDiseaseorAttack", "Includes coronary heart disease or prior heart attack.")
                render_binary_choice("Cholesterol check", "CholCheck", "Confirms whether cholesterol has been checked recently.")
                render_binary_choice("Has healthcare coverage", "AnyHealthcare", "Access to regular care influences screening and prevention opportunities.")

        with st.expander("Lifestyle", expanded=True):
            col_a, col_b = st.columns(2)
            with col_a:
                render_binary_choice("Current smoker", "Smoker", "Select yes if the patient currently smokes.")
                render_binary_choice("Heavy alcohol use", "HvyAlcoholConsump", "Includes heavy alcohol consumption patterns.")
                render_binary_choice("Regular physical activity", "PhysActivity", "This helps capture an active lifestyle.")
            with col_b:
                render_binary_choice("Regular fruit intake", "Fruits", "Includes regular fruit consumption.")
                render_binary_choice("Regular vegetable intake", "Veggies", "Includes regular vegetable consumption.")

        with st.expander("Health and Wellbeing", expanded=True):
            col_a, col_b = st.columns(2)
            with col_a:
                render_choice_dropdown("General health", "GenHlth", GENERAL_HEALTH_OPTIONS, "Overall wellness and health perception.")
                render_choice_dropdown("Education", "Education", EDUCATION_OPTIONS, "Highest completed education level.")
            with col_b:
                render_choice_dropdown("Income bracket", "Income", INCOME_OPTIONS, "Estimated financial bracket.")
                render_binary_choice("Cost barrier to care", "NoDocbcCost", "Select yes if cost prevents the patient from seeing a doctor.")

        with st.expander("Symptom and Functioning", expanded=True):
            col_a, col_b = st.columns(2)
            with col_a:
                st.number_input("Mental health days (past 30 days)", min_value=0, max_value=30, value=int(st.session_state.get("MentHlth", 0)), step=1, key="MentHlth", help="Number of days with poor mental health.")
                st.number_input("Physical health days (past 30 days)", min_value=0, max_value=30, value=int(st.session_state.get("PhysHlth", 0)), step=1, key="PhysHlth", help="Number of days with poor physical health.")
            with col_b:
                render_binary_choice("Difficulty walking", "DiffWalk", "Indicates whether walking is difficult.")

        submitted = st.form_submit_button("Predict risk", use_container_width=True)

    if submitted:
        backend_url = st.session_state.get("backend_url", DEFAULT_BACKEND_URL)
        if not ping_backend(backend_url):
            st.error("The backend is not reachable right now. Please start the FastAPI service and try again.")
            return

        with st.spinner("Analyzing patient profile..."):
            payload = build_patient_payload()
            result = post_json(f"{backend_url}/api/v1/full-analysis", payload)

        st.balloons()
        st.success("Analysis completed successfully")

        prediction = result.get("prediction", {})
        probability = float(prediction.get("probability", 0.0))
        probability_pct = f"{probability:.2%}"
        risk_level = prediction.get("risk_level", "Unknown")
        confidence = "High" if probability >= 0.5 else "Low"

        st.markdown("### Risk overview")
        risk_col, detail_col = st.columns([1.1, 1.2])
        with risk_col:
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Risk score"},
                    gauge={
                        "axis": {"range": [None, 100], "tickwidth": 1},
                        "bar": {"color": "#ef4444" if risk_level == "High" else "#3b82f6"},
                        "steps": [
                            {"range": [0, 50], "color": "#e0f2fe"},
                            {"range": [50, 80], "color": "#fef3c7"},
                            {"range": [80, 100], "color": "#fee2e2"},
                        ],
                    },
                )
            )
            fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with detail_col:
            st.metric("Risk probability", probability_pct)
            st.metric("Risk category", risk_level)
            st.metric("Confidence", confidence)
            st.metric("Prediction", prediction.get("prediction", "Unknown"))

        st.markdown("### Clinical interpretation")
        col_a, col_b = st.columns(2)
        with col_a:
            with st.container():
                st.markdown("#### SHAP summary")
                explanation = result.get("explanation", {}) or {}
                top_risk = explanation.get("top_risk_factors", [])[:3]
                top_protective = explanation.get("top_protective_factors", [])[:3]
                if top_risk:
                    for item in top_risk:
                        st.write(f"• {item.get('feature', 'Unknown')}: {item.get('shap_value', 0):+.2f}")
                else:
                    st.info("No SHAP risk factors were returned for this profile.")
                if top_protective:
                    st.caption("Protective signals")
                    for item in top_protective:
                        st.write(f"• {item.get('feature', 'Unknown')}: {item.get('shap_value', 0):+.2f}")
        with col_b:
            with st.container():
                st.markdown("#### Recommendations")
                recommendation_payload = result.get("recommendation", {}) or {}
                actions = recommendation_payload.get("priority_actions", [])
                if actions:
                    for action in actions:
                        st.write(f"• **{action.get('title', 'Action')}** — {action.get('reason', '')}")
                else:
                    st.info("No personalized actions were generated for this case.")

        with st.expander("Clinical summary", expanded=True):
            st.write(result.get("llm_explanation", {}).get("summary", "No summary available."))

        with st.expander("Patient guidance", expanded=True):
            st.write(result.get("llm_explanation", {}).get("recommendation_explanation", "No explanation available."))


def render_explainability() -> None:
    st.title("Explainability")
    st.write("SHAP-driven reasoning for the latest patient analysis.")
    if st.button("Load explainability report"):
        with st.spinner("Loading SHAP insights..."):
            payload = build_patient_payload()
            result = post_json(f"{st.session_state.backend_url}/api/v1/explain", payload)
        st.success("SHAP explanation loaded")
        st.metric("Prediction", result["prediction"])
        st.metric("Probability", f"{result['probability']:.2%}")
        st.metric("Risk category", result["risk_level"])

        cols = st.columns(2)
        with cols[0]:
            st.subheader("Top contributors")
            risk_df = pd.DataFrame(result.get("top_risk_factors", []))
            if not risk_df.empty:
                st.dataframe(risk_df[["feature", "shap_value"]], use_container_width=True)
            else:
                st.info("No top risk contributors available.")
        with cols[1]:
            st.subheader("Protective factors")
            protect_df = pd.DataFrame(result.get("top_protective_factors", []))
            if not protect_df.empty:
                st.dataframe(protect_df[["feature", "shap_value"]], use_container_width=True)
            else:
                st.info("No protective factors available.")


def render_recommendations() -> None:
    st.title("Recommendations")
    st.write("Personalized next steps generated from the backend recommendation engine.")
    if st.button("Load recommendations"):
        with st.spinner("Preparing recommendations..."):
            payload = build_patient_payload()
            result = post_json(f"{st.session_state.backend_url}/api/v1/recommend", payload)
        st.success("Recommendations generated")

        st.subheader("Priority actions")
        actions = result.get("priority_actions", [])
        if actions:
            for action in actions:
                st.markdown(f"- **{action['title']}** ({action['priority']}) — {action['reason']}")
        else:
            st.info("No priority actions were generated for this profile.")

        st.subheader("Lifestyle guidance")
        for category, entries in result.get("recommendations", {}).items():
            with st.expander(category.replace("_", " ").title()):
                for entry in entries:
                    st.write(f"- {entry['recommendation']}")

        st.subheader("Positive observations")
        for item in result.get("positive_observations", []):
            st.write(f"- {item}")

        st.subheader("Medical disclaimer")
        st.warning(result.get("disclaimer", "No disclaimer available."))


def render_analytics() -> None:
    st.title("Analytics")
    st.write("Session-based insights and backend metrics.")

    try:
        model_info = fetch_json(f"{st.session_state.backend_url}/api/v1/model-info")
    except Exception:
        model_info = {}

    if model_info:
        st.metric("Model", model_info.get("model_name", "Unknown"))
        st.metric("Feature count", model_info.get("feature_count", 0))

    history = st.session_state.get("history", [])
    if history:
        history_df = pd.DataFrame(history)
        fig = px.line(history_df, x=history_df.index, y="probability", markers=True, title="Prediction history")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No prediction history yet. Run a prediction to populate the chart.")

    if st.button("Record current analysis"):
        payload = build_patient_payload()
        try:
            result = post_json(f"{st.session_state.backend_url}/api/v1/predict", payload)
            history = st.session_state.setdefault("history", [])
            history.append({"probability": result.get("probability", 0), "risk_level": result.get("risk_level", "Unknown")})
            st.session_state.history = history
            st.success("Prediction history updated")
        except Exception as exc:
            st.error(f"Unable to record prediction: {exc}")


def render_about() -> None:
    st.title("About")
    st.write("This portfolio-ready dashboard demonstrates how a modern healthcare AI application can expose predictions and recommendations through a clean, API-first interface.")
    st.markdown("- Built with Streamlit and Plotly\n- Consumes FastAPI endpoints\n- Reuses existing ML and explainability modules\n- Designed for healthcare product demos and interviews")


def main() -> None:
    st.session_state.setdefault("backend_url", DEFAULT_BACKEND_URL)
    page = render_sidebar()

    if page == "Home":
        render_home()
    elif page == "Prediction":
        render_prediction()
    elif page == "Explainability":
        render_explainability()
    elif page == "Recommendations":
        render_recommendations()
    elif page == "Analytics":
        render_analytics()
    else:
        render_about()


if __name__ == "__main__":
    main()
