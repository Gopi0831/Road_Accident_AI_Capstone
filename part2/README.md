# Part 2 - Road Accident Severity Prediction

## Overview

In this part of the project, I built classification models to predict road collision severity.

The target column is `collision_severity`, which contains three classes:

- 1 - Fatal
- 2 - Serious
- 3 - Slight

## Features Used

I selected the following features for model training:

- number_of_vehicles
- number_of_casualties
- day_of_week
- road_type
- speed_limit
- junction_detail
- light_conditions
- weather_conditions
- road_surface_conditions
- urban_or_rural_area

Columns directly related to adjusted or injury-based severity were not used as model features because they may introduce data leakage.

## Train and Test Split

The dataset was split into 80% training data and 20% testing data.

A stratified split was used so that the collision severity class distribution was maintained in both sets.

## Baseline Model

A Decision Tree classifier with a maximum depth of 5 was used as the baseline model.

The baseline model achieved a test accuracy of 75.96%.

However, the classification report showed that the model failed to correctly identify fatal collisions. Most records were predicted as slight collisions.

This showed that accuracy alone was not a suitable metric for this imbalanced dataset.

## Random Forest Model

I used a Random Forest classifier as the ensemble model.

Balanced class weights were used because the dataset contains many more slight collisions than fatal collisions.

The Random Forest achieved a test accuracy of 54.89%. Its overall accuracy was lower than the baseline, but it improved the detection of minority severity classes.

The fatal collision recall increased from 0.00 in the baseline model to 0.49.

## Model Tuning

GridSearchCV was used to tune the Random Forest model.

The model was evaluated using macro F1 score because macro F1 gives importance to all severity classes instead of focusing mainly on the majority class.

The best parameters were:

- max_depth: 15
- min_samples_split: 5
- n_estimators: 100

The tuned model achieved:

- Test accuracy: 56.57%
- Macro F1 score: 0.3674

The tuned model gave a better macro F1 score than the baseline and improved the prediction of serious and fatal collision classes.

## Model Selection

The tuned Random Forest was selected as the final model.

Although the Decision Tree had higher overall accuracy, it failed to detect fatal collisions in the test set. The tuned Random Forest provided a better balance across the three collision severity classes.

## How to Run

Install the required libraries:

```bash
pip install pandas scikit-learn joblib