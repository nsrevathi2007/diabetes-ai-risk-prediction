# Global SHAP Explainability Report

## Model Behavior
Global SHAP analysis estimates how each feature contributes to diabetes risk predictions across the sampled dataset.

## Top 20 Features
| Rank | Feature | Mean Absolute SHAP |
| ---: | --- | ---: |
| 1 | GenHlth | 0.631750 |
| 2 | HighBP | 0.519986 |
| 3 | Age | 0.428762 |
| 4 | BMI | 0.395523 |
| 5 | HighChol | 0.312209 |
| 6 | Income | 0.133699 |
| 7 | Sex | 0.131660 |
| 8 | CholCheck | 0.089137 |
| 9 | HvyAlcoholConsump | 0.071893 |
| 10 | MentHlth | 0.065305 |
| 11 | HeartDiseaseorAttack | 0.061181 |
| 12 | Education | 0.049649 |
| 13 | DiffWalk | 0.045291 |
| 14 | Fruits | 0.021884 |
| 15 | PhysHlth | 0.018824 |
| 16 | PhysActivity | 0.017711 |
| 17 | Stroke | 0.014966 |
| 18 | Smoker | 0.012189 |
| 19 | Veggies | 0.007643 |
| 20 | AnyHealthcare | 0.005908 |

## Generated Artifacts
- Summary plot: `reports\shap\summary_plot.png`
- Bar plot: `reports\shap\bar_plot.png`
- Feature importance: `reports\shap\feature_importance.csv`