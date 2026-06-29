# Optimization Report

## Summary
The best optimized model was CatBoost with ROC-AUC 0.8278, recall 0.8687, and threshold 0.10.

Percentage improvement is calculated from baseline ROC-AUC to optimized ROC-AUC.

## Optimized Comparison
| Model | Baseline ROC-AUC | Optimized ROC-AUC | Baseline Recall | Optimized Recall | Improvement |
| --- | ---: | ---: | ---: | ---: | ---: |
| RandomForest | 0.5731 | 0.8212 | 0.1757 | 0.9641 | 43.30% |
| XGBoost | 0.5753 | 0.8275 | 0.1767 | 0.8642 | 43.84% |
| LightGBM | 0.5732 | 0.8273 | 0.1679 | 0.8652 | 44.33% |
| CatBoost | 0.5745 | 0.8278 | 0.1730 | 0.8687 | 44.10% |

## Selected Strategies
- RandomForest: RandomUnderSampler imbalance handling, threshold 0.23, hyperparameters `{"max_depth": 9, "min_samples_leaf": 1, "min_samples_split": 7, "n_estimators": 244}`
- XGBoost: Original imbalance handling, threshold 0.10, hyperparameters `{"colsample_bytree": 0.7368871719068161, "learning_rate": 0.0983175470436899, "max_depth": 4, "n_estimators": 177, "subsample": 0.986324594883905}`
- LightGBM: Original imbalance handling, threshold 0.10, hyperparameters `{"colsample_bytree": 0.768320974384542, "learning_rate": 0.03640990946107894, "n_estimators": 184, "num_leaves": 29, "subsample": 0.9727540258561057}`
- CatBoost: Original imbalance handling, threshold 0.10, hyperparameters `{"depth": 7, "iterations": 276, "l2_leaf_reg": 2.254983018416771, "learning_rate": 0.045988940028932494}`