# Optimization Report

## Summary
The best optimized model was LightGBM with ROC-AUC 0.8269, recall 0.8626, and threshold 0.10.

Percentage improvement is calculated from baseline ROC-AUC to optimized ROC-AUC.

## Optimized Comparison
| Model | Baseline ROC-AUC | Optimized ROC-AUC | Baseline Recall | Optimized Recall | Improvement |
| --- | ---: | ---: | ---: | ---: | ---: |
| RandomForest | 0.5731 | 0.8230 | 0.1757 | 0.8857 | 43.61% |
| XGBoost | 0.5753 | 0.8257 | 0.1767 | 0.8686 | 43.52% |
| LightGBM | 0.5732 | 0.8269 | 0.1679 | 0.8626 | 44.27% |
| CatBoost | 0.5745 | 0.8265 | 0.1730 | 0.8704 | 43.88% |

## Selected Strategies
- RandomForest: Original imbalance handling, threshold 0.10, hyperparameters `{"max_depth": 11, "min_samples_leaf": 6, "min_samples_split": 2, "n_estimators": 104}`
- XGBoost: Original imbalance handling, threshold 0.10, hyperparameters `{"colsample_bytree": 0.9124217733388136, "learning_rate": 0.13394334706750485, "max_depth": 3, "n_estimators": 64, "subsample": 0.8803345035229626}`
- LightGBM: Original imbalance handling, threshold 0.10, hyperparameters `{"colsample_bytree": 0.9124217733388136, "learning_rate": 0.13394334706750485, "n_estimators": 64, "num_leaves": 17, "subsample": 0.8803345035229626}`
- CatBoost: Original imbalance handling, threshold 0.10, hyperparameters `{"depth": 6, "iterations": 84, "l2_leaf_reg": 5.190609389379256, "learning_rate": 0.08960785365368121}`