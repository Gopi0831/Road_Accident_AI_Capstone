# Part 2 – Predictive Modeling and Statistical Analysis

## Overview

This part of the Road Accident AI Capstone focuses on predictive modeling and statistical evaluation of road accident data.

The analysis includes regression modeling, classification, class imbalance handling, decision threshold sensitivity, regularization comparison, and bootstrap analysis.

## Models Implemented

### 1. Linear Regression

Linear Regression was used to predict the number of casualties.

Results:

- Mean Squared Error (MSE): 0.446495
- R² Score: 0.099213

The three features with the highest absolute coefficients were:

1. `location_northing_osgr`
2. `latitude`
3. `collision_severity_3`

### 2. Ridge Regression

Ridge Regression was applied to evaluate whether regularization improved regression performance.

Results:

- Mean Squared Error (MSE): 0.446472
- R² Score: 0.099260

The Ridge Regression results were very similar to Linear Regression, with a small improvement in MSE and R² score.

## Class Distribution

The binary classification target showed an imbalanced class distribution.

Training class distribution:

- Class 0: 67,973 samples (81.50%)
- Class 1: 15,433 samples (18.50%)

To handle class imbalance, balanced class weights were automatically applied using Logistic Regression.

## Logistic Regression

A Logistic Regression model with `C=1.0` was trained.

### Performance Metrics

- Accuracy: 0.6685
- Precision: 0.3152
- Recall: 0.6692
- F1 Score: 0.4286
- AUC: 0.7271

The model achieved relatively high recall for the positive class, identifying approximately 67% of positive cases.

## Decision Threshold Sensitivity

Different classification thresholds were evaluated.

| Threshold | Precision | Recall | F1 Score |
|-----------|-----------|--------|----------|
| 0.3 | 0.2301 | 0.9174 | 0.3679 |
| 0.4 | 0.2679 | 0.8218 | 0.4040 |
| 0.5 | 0.3152 | 0.6692 | 0.4286 |
| 0.6 | 0.3714 | 0.4815 | 0.4193 |
| 0.7 | 0.4324 | 0.2724 | 0.3342 |

Among the evaluated thresholds, `0.5` produced the maximum F1 score.

## Regularization Comparison

Two Logistic Regression regularization settings were compared.

| Model | Precision | Recall | AUC |
|-------|-----------|--------|-----|
| Logistic C=1.0 | 0.3152 | 0.6692 | 0.7271 |
| Logistic C=0.01 | 0.3161 | 0.6695 | 0.7274 |

The stronger regularization model (`C=0.01`) produced a very small improvement in AUC.

## Bootstrap AUC Difference

Bootstrap analysis was performed using 500 samples to evaluate the AUC difference between the Logistic Regression models.

Results:

- Mean AUC Difference: -0.000231
- 2.5th Percentile: -0.000530
- 97.5th Percentile: 0.000074

The 95% confidence interval includes zero. Therefore, the difference in AUC between the two Logistic Regression models is not statistically significant.

## ROC Curve

The ROC curve generated during model evaluation is available at:

`images/roc_curve.png`

## Conclusion

The regression models showed limited predictive power for estimating the exact number of casualties, with R² scores of approximately 0.10.

For binary classification, Logistic Regression achieved an AUC of approximately 0.727. The model demonstrated useful recall for identifying positive cases.

Threshold analysis showed the trade-off between precision and recall, while bootstrap analysis indicated that stronger regularization did not produce a statistically significant improvement in AUC.

Part 2 successfully demonstrates regression, classification, imbalance handling, threshold analysis, regularization, and statistical model comparison.