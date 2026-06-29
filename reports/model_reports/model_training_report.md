# Model Training Report

## Model Comparison
| rank | model_name | roc_auc | recall | f1 | accuracy |
| --- | --- | --- | --- | --- | --- |
| 1 | DecisionTree | 0.6028069716907356 | 0.33243740274437683 | 0.3142341378618707 | 0.7978358561967833 |
| 2 | XGBoost | 0.5753244869073572 | 0.176686942990522 | 0.2641988365943945 | 0.8628784295175024 |
| 3 | CatBoost | 0.574458746501088 | 0.17300891215164804 | 0.2617722602739726 | 0.864041311888994 |
| 4 | LightGBM | 0.5731834001230451 | 0.16791625406705332 | 0.25812765032075674 | 0.8655195521917376 |
| 5 | RandomForest | 0.5730660220533802 | 0.17569670391851747 | 0.25869610497812956 | 0.8597051403342794 |
| 6 | LogisticRegression | 0.5672077670684438 | 0.1584382515207243 | 0.2424767265641914 | 0.8620703248186692 |

## Best Performing Model
The best baseline model was DecisionTree based on ROC-AUC, recall, F1 score, and accuracy.

## Healthcare Considerations
Healthcare classification should prioritize ROC-AUC, recall, and F1 score because false negatives can delay treatment.