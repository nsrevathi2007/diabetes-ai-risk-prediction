# LLM Integration Report

## Architecture
Recommendation Engine -> Structured Recommendation JSON -> Prompt Builder -> LLM Provider or Template Mode -> Structured Explanation JSON.

The recommendation engine remains the source of truth. The LLM layer only explains existing predictions, SHAP factors, observations, and recommendations.

## Prompt Design
Prompts are split into system and user prompts. The user prompt includes only whitelisted recommendation fields and avoids duplicate SHAP detail.

## Offline Mode
Template Mode generated 10 explanation outputs. LLM mode generated 0 explanation outputs.

## Safety
The system prompt prohibits diagnosis, prescriptions, replacing healthcare professionals, and invented recommendations. The safety layer removes prompt-injection text and rejects unsupported generated priority actions.

## Example Outputs
```json
{
  "patient_id": "patient_01",
  "risk_level": "Low Risk",
  "summary": "The model estimated this patient's diabetes risk as Low Risk with a prediction probability of 0.0354. This explanation summarizes the existing structured recommendation output.",
  "why_this_prediction": "The main SHAP factors increasing the prediction were GenHlth, Income, Education, CholCheck, HvyAlcoholConsump. Protective SHAP factors reducing the prediction were BMI, HighBP, HighChol, MentHlth, Sex.",
  "positive_observations": [
    "Fruit consumption is a positive nutrition signal.",
    "Vegetable consumption is a positive nutrition signal.",
    "Not smoking is a positive health indicator.",
    "Avoiding heavy alcohol consumption is a positive health indicator.",
    "No high blood pressure indicator was observed in the patient data.",
    "No high cholesterol indicator was observed in the patient data.",
    "BMI appears protective in this prediction.",
    "HighBP appears protective in this prediction.",
    "HighChol appears protective in this prediction.",
    "MentHlth appears protective in this prediction.",
    "Sex appears protective in this prediction."
  ],
  "priority_actions": [
    {
      "title": "Maintain protective routines",
      "priority": "High",
      "explanation": "GenHlth was an important contributor to the prediction."
    },
    {
      "title": "Maintain protective routines",
      "priority": "High",
      "explanation": "Income was an important contributor to the prediction."
    },
    {
      "title": "Plan healthcare follow-up",
      "priority": "High",
      "explanation": "GenHlth was an important contributor to the prediction."
    },
    {
      "title": "Maintain protective routines",
      "priority": "Low",
      "explanation": "Education was an important contributor to the prediction."
    },
    {
      "title": "Stay current with preventive screening",
      "priority": "Low",
      "explanation": "CholCheck was an important contributor to the prediction."
    }
  ],
  "recommendation_explanation": "The structured recommendation engine produced guidance in these categories: general wellness, follow up, preventive screening. These items are explanatory and educational only.",
  "next_steps": "Continue protective lifestyle habits and routine preventive care.",
  "disclaimer": "This information is generated for educational purposes only and should not be considered medical advice. Please consult a qualified healthcare professional for diagnosis or treatment.",
  "generated_at": "2026-06-30T13:29:40.711501Z",
  "generation_mode": "template",
  "model": "gpt-4o-mini",
  "source_recommendation_count": 5
}
```

## Future Improvements
- Add provider adapters for Azure OpenAI, Ollama, Gemini, and other OpenAI-compatible services.
- Persist provider latency and fallback metrics for monitoring.
- Add frontend controls for explanation tone and detail level while preserving the same structured source JSON.

## Disclaimer
This information is generated for educational purposes only and should not be considered medical advice. Please consult a qualified healthcare professional for diagnosis or treatment.