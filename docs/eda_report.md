# Exploratory Data Analysis Report
**Generated:** 2026-06-29 19:41:07

## Table of Contents

1. [Dataset Overview](#dataset-overview)
2. [Target Variable Analysis](#target-variable-analysis)
3. [Feature Analysis](#feature-analysis)
4. [Correlation Analysis](#correlation-analysis)
5. [Healthcare Insights](#healthcare-insights)
6. [Feature Ranking](#feature-ranking)
7. [Recommendations](#recommendations)

## Dataset Overview

- **Number of Rows:** 253,680
- **Number of Columns:** 22
- **Memory Usage:** 42.58 MB
- **Duplicate Rows:** 24206 (9.54%)
- **Missing Values:** 0 (0.00%)

### Descriptive Statistics

| Feature | Count | Mean | Median | Std | Min | Max |
|---------|-------|------|--------|-----|-----|-----|
| Diabetes_binary | 253680 | 0.14 | 0.00 | 0.35 | 0.00 | 1.00 |
| HighBP | 253680 | 0.43 | 0.00 | 0.49 | 0.00 | 1.00 |
| HighChol | 253680 | 0.42 | 0.00 | 0.49 | 0.00 | 1.00 |
| CholCheck | 253680 | 0.96 | 1.00 | 0.19 | 0.00 | 1.00 |
| BMI | 253680 | 28.38 | 27.00 | 6.61 | 12.00 | 98.00 |
| Smoker | 253680 | 0.44 | 0.00 | 0.50 | 0.00 | 1.00 |
| Stroke | 253680 | 0.04 | 0.00 | 0.20 | 0.00 | 1.00 |
| HeartDiseaseorAttack | 253680 | 0.09 | 0.00 | 0.29 | 0.00 | 1.00 |
| PhysActivity | 253680 | 0.76 | 1.00 | 0.43 | 0.00 | 1.00 |
| Fruits | 253680 | 0.63 | 1.00 | 0.48 | 0.00 | 1.00 |
| Veggies | 253680 | 0.81 | 1.00 | 0.39 | 0.00 | 1.00 |
| HvyAlcoholConsump | 253680 | 0.06 | 0.00 | 0.23 | 0.00 | 1.00 |
| AnyHealthcare | 253680 | 0.95 | 1.00 | 0.22 | 0.00 | 1.00 |
| NoDocbcCost | 253680 | 0.08 | 0.00 | 0.28 | 0.00 | 1.00 |
| GenHlth | 253680 | 2.51 | 2.00 | 1.07 | 1.00 | 5.00 |
| MentHlth | 253680 | 3.18 | 0.00 | 7.41 | 0.00 | 30.00 |
| PhysHlth | 253680 | 4.24 | 0.00 | 8.72 | 0.00 | 30.00 |
| DiffWalk | 253680 | 0.17 | 0.00 | 0.37 | 0.00 | 1.00 |
| Sex | 253680 | 0.44 | 0.00 | 0.50 | 0.00 | 1.00 |
| Age | 253680 | 8.03 | 8.00 | 3.05 | 1.00 | 13.00 |
| Education | 253680 | 5.05 | 5.00 | 0.99 | 1.00 | 6.00 |
| Income | 253680 | 6.05 | 7.00 | 2.07 | 1.00 | 8.00 |

## Target Variable Analysis

### Class Distribution

| Class | Count | Percentage |
|-------|-------|-----------|
| 0 | 218,334 | 86.07% |
| 1 | 35,346 | 13.93% |

### Class Imbalance Ratio

- **Ratio:** 6.18:1
- **Majority Class:** 0 (218,334 samples)
- **Minority Class:** 1 (35,346 samples)

**Note:** Class imbalance detected. Consider using techniques like SMOTE, class weights, or stratified sampling during model training.

## Feature Analysis

- **Total Features:** 22
- **Numerical Features:** 22
- **Categorical Features:** 0

### Feature Summary

| Feature | Type | Count | Unique | Mean/Top |
|---------|------|-------|--------|----------|
| Age | Numerical | 253680 | 13 | 8.03 |
| AnyHealthcare | Numerical | 253680 | 2 | 0.95 |
| BMI | Numerical | 253680 | 84 | 28.38 |
| CholCheck | Numerical | 253680 | 2 | 0.96 |
| Diabetes_binary | Numerical | 253680 | 2 | 0.14 |
| DiffWalk | Numerical | 253680 | 2 | 0.17 |
| Education | Numerical | 253680 | 6 | 5.05 |
| Fruits | Numerical | 253680 | 2 | 0.63 |
| GenHlth | Numerical | 253680 | 5 | 2.51 |
| HeartDiseaseorAttack | Numerical | 253680 | 2 | 0.09 |
| HighBP | Numerical | 253680 | 2 | 0.43 |
| HighChol | Numerical | 253680 | 2 | 0.42 |
| HvyAlcoholConsump | Numerical | 253680 | 2 | 0.06 |
| Income | Numerical | 253680 | 8 | 6.05 |
| MentHlth | Numerical | 253680 | 31 | 3.18 |
| NoDocbcCost | Numerical | 253680 | 2 | 0.08 |
| PhysActivity | Numerical | 253680 | 2 | 0.76 |
| PhysHlth | Numerical | 253680 | 31 | 4.24 |
| Sex | Numerical | 253680 | 2 | 0.44 |
| Smoker | Numerical | 253680 | 2 | 0.44 |
| Stroke | Numerical | 253680 | 2 | 0.04 |
| Veggies | Numerical | 253680 | 2 | 0.81 |

## Correlation Analysis

### Top Positive Correlations with Target

- **GenHlth:** 0.2936
- **HighBP:** 0.2631
- **DiffWalk:** 0.2183
- **BMI:** 0.2168
- **HighChol:** 0.2003

### Top Negative Correlations with Target

- **Income:** -0.1639
- **Education:** -0.1245
- **PhysActivity:** -0.1181
- **HvyAlcoholConsump:** -0.0571
- **Veggies:** -0.0566

### Strongest Feature-to-Feature Correlations

- **GenHlth - PhysHlth:** 0.5244
- **PhysHlth - DiffWalk:** 0.4784
- **GenHlth - DiffWalk:** 0.4569
- **Education - Income:** 0.4491
- **GenHlth - Income:** -0.3700

## Healthcare Insights

Key health indicators have been visualized against the target variable:

- **BMI vs Diabetes**
- **Age vs Diabetes**
- **High Blood Pressure vs Diabetes**
- **High Cholesterol vs Diabetes**
- **Physical Activity vs Diabetes**
- **Smoking vs Diabetes**
- **Heart Disease/Attack vs Diabetes**
- **General Health vs Diabetes**
- **Income vs Diabetes**
- **Education vs Diabetes**

These visualizations reveal the relationship between health indicators and diabetes risk.

## Feature Ranking

### Top 10 Features by Mutual Information

| Rank | Feature | MI Score |
|------|---------|----------|
| 1 | HighBP | 0.0531 |
| 2 | GenHlth | 0.0525 |
| 3 | PhysActivity | 0.0524 |
| 4 | AnyHealthcare | 0.0506 |
| 5 | CholCheck | 0.0488 |
| 6 | Fruits | 0.0423 |
| 7 | Veggies | 0.0422 |
| 8 | HighChol | 0.0407 |
| 9 | BMI | 0.0308 |
| 10 | Sex | 0.0257 |

## Recommendations Before Preprocessing

### Data Quality

- ✓ Verify that duplicate rows are intentional or remove if needed
- ✓ Investigate any missing values and decide on handling strategy
- ✓ Check for outliers and decide on treatment strategy

### Feature Engineering

- ✓ Consider creating interaction terms for highly correlated features
- ✓ Evaluate polynomial features for non-linear relationships
- ✓ Consider domain-specific feature engineering based on healthcare knowledge

### Class Imbalance

- ✓ Class imbalance detected - consider SMOTE, class weights, or stratification
- ✓ Use appropriate metrics (precision, recall, F1) instead of accuracy

### Feature Selection

- ✓ Use mutual information scores to guide feature selection
- ✓ Consider removing features with near-zero variance
- ✓ Evaluate multicollinearity and remove highly correlated features if needed

### Modeling

- ✓ Use stratified k-fold cross-validation
- ✓ Consider ensemble methods to handle class imbalance
- ✓ Use appropriate evaluation metrics for imbalanced classification
