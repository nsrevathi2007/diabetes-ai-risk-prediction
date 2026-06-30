# Recommendation Engine Report

## Summary
The Phase 7 recommendation engine generated informational, non-diagnostic recommendations from existing model predictions, saved SHAP explanations, and patient health indicators.

## SHAP Artifact Strategy
Structured SHAP JSON artifacts are preferred when present. Existing SHAP markdown reports are parsed as a fallback, so SHAP values are not regenerated.

## Risk Level Counts
- Low Risk: 9
- Moderate Risk: 1

## Generated Patients
| Patient | Risk Level | Probability | Priority Actions |
| --- | --- | ---: | ---: |
| patient_01 | Low Risk | 0.0354 | 5 |
| patient_02 | Low Risk | 0.2182 | 5 |
| patient_03 | Low Risk | 0.0071 | 3 |
| patient_04 | Low Risk | 0.0031 | 2 |
| patient_05 | Low Risk | 0.2200 | 5 |
| patient_06 | Moderate Risk | 0.3982 | 5 |
| patient_07 | Low Risk | 0.0355 | 5 |
| patient_08 | Low Risk | 0.1870 | 5 |
| patient_09 | Low Risk | 0.0340 | 5 |
| patient_10 | Low Risk | 0.0680 | 5 |

## Disclaimer
This information is generated for educational purposes only and should not be considered medical advice. Please consult a qualified healthcare professional for diagnosis or treatment.