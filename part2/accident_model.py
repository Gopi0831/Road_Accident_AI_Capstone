import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


os.makedirs("part2/images", exist_ok=True)


# Load the cleaned dataset created in Part 1
df = pd.read_csv("part1/cleaned_data.csv")

print("===== DATASET =====")
print("Dataset shape:", df.shape)


# Regression target
y_reg = df["number_of_casualties"]

# Binary classification target
casualty_median = y_reg.median()
y_clf = (y_reg > casualty_median).astype(int)

print("\nRegression label: number_of_casualties")
print("Classification rule:")
print("1 = number_of_casualties > median")
print("0 = number_of_casualties <= median")
print("Median casualty value:", casualty_median)


# Remove targets and columns that can leak target information
drop_columns = [
    "number_of_casualties",
    "collision_index",
    "collision_ref_no",
    "date",
    "time",
    "lsoa_of_accident_location",
    "enhanced_severity_collision",
    "collision_injury_based",
    "collision_adjusted_severity_serious",
    "collision_adjusted_severity_slight",
]

X = df.drop(columns=drop_columns, errors="ignore")


# Treat coded accident fields as categorical features
categorical_columns = [
    "police_force",
    "collision_severity",
    "day_of_week",
    "local_authority_district",
    "local_authority_ons_district",
    "local_authority_highway",
    "local_authority_highway_current",
    "first_road_class",
    "road_type",
    "junction_detail_historic",
    "junction_detail",
    "junction_control",
    "second_road_class",
    "pedestrian_crossing_human_control_historic",
    "pedestrian_crossing_physical_facilities_historic",
    "pedestrian_crossing",
    "light_conditions",
    "weather_conditions",
    "road_surface_conditions",
    "special_conditions_at_site",
    "carriageway_hazards_historic",
    "carriageway_hazards",
    "urban_or_rural_area",
    "did_police_officer_attend_scene_of_accident",
    "trunk_road_flag",
]

categorical_columns = [
    col for col in categorical_columns if col in X.columns
]

X[categorical_columns] = X[categorical_columns].astype("category")


# Fill any remaining categorical missing values
for col in categorical_columns:
    if X[col].isnull().sum() > 0:
        mode_value = X[col].mode()[0]
        X[col] = X[col].fillna(mode_value)


# Fill any remaining numeric missing values using training-safe logic later
numeric_columns = X.select_dtypes(include=np.number).columns


# One-hot encode nominal categorical columns
X = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=True,
    dtype=int,
)

print("\nFeature matrix shape after one-hot encoding:")
print(X.shape)


# Use one common split so regression and classification use the same rows
indices = np.arange(len(X))

train_indices, test_indices = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
)

X_train = X.iloc[train_indices].copy()
X_test = X.iloc[test_indices].copy()

y_reg_train = y_reg.iloc[train_indices]
y_reg_test = y_reg.iloc[test_indices]

y_clf_train = y_clf.iloc[train_indices]
y_clf_test = y_clf.iloc[test_indices]


# Fill numeric missing values using training medians only
for col in X_train.columns:
    if X_train[col].isnull().sum() > 0:
        train_median = X_train[col].median()
        X_train[col] = X_train[col].fillna(train_median)
        X_test[col] = X_test[col].fillna(train_median)


print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)


# Scaling without data leakage
scaler = StandardScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Linear Regression
print("\n===== LINEAR REGRESSION =====")

linear_model = LinearRegression()

linear_model.fit(X_train_scaled, y_reg_train)

y_pred_reg = linear_model.predict(X_test_scaled)

linear_mse = mean_squared_error(y_reg_test, y_pred_reg)
linear_r2 = r2_score(y_reg_test, y_pred_reg)

print("MSE:", linear_mse)
print("R2:", linear_r2)


coefficient_table = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": linear_model.coef_,
})

coefficient_table["Absolute Coefficient"] = (
    coefficient_table["Coefficient"].abs()
)

coefficient_table = coefficient_table.sort_values(
    "Absolute Coefficient",
    ascending=False,
)

print("\nLinear Regression coefficients:")
print(coefficient_table.to_string(index=False))

print("\nTop 3 absolute coefficients:")
print(coefficient_table.head(3).to_string(index=False))


# Ridge Regression
print("\n===== RIDGE REGRESSION =====")

ridge_model = Ridge(alpha=1.0)

ridge_model.fit(X_train_scaled, y_reg_train)

ridge_predictions = ridge_model.predict(X_test_scaled)

ridge_mse = mean_squared_error(
    y_reg_test,
    ridge_predictions,
)

ridge_r2 = r2_score(
    y_reg_test,
    ridge_predictions,
)

regression_comparison = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Ridge Regression",
    ],
    "MSE": [
        linear_mse,
        ridge_mse,
    ],
    "R2": [
        linear_r2,
        ridge_r2,
    ],
})

print("\nRegression comparison:")
print(regression_comparison.to_string(index=False))


# Classification class distribution
print("\n===== CLASS DISTRIBUTION =====")

class_counts_before = y_clf_train.value_counts().sort_index()

print("Training class counts before handling imbalance:")
print(class_counts_before)

class_percentages = (
    y_clf_train.value_counts(normalize=True).sort_index() * 100
)

print("\nTraining class percentages:")
print(class_percentages)


# Logistic Regression with balanced class weights
print("\n===== LOGISTIC REGRESSION C=1.0 =====")

logistic_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    C=1.0,
)

logistic_model.fit(
    X_train_scaled,
    y_clf_train,
)

print("\nClass counts after imbalance handling:")
print("Class weights are balanced automatically by LogisticRegression.")
print(class_counts_before)

y_pred_clf = logistic_model.predict(X_test_scaled)

y_proba_clf = logistic_model.predict_proba(
    X_test_scaled
)[:, 1]

baseline_accuracy = accuracy_score(
    y_clf_test,
    y_pred_clf,
)

baseline_precision = precision_score(
    y_clf_test,
    y_pred_clf,
    zero_division=0,
)

baseline_recall = recall_score(
    y_clf_test,
    y_pred_clf,
    zero_division=0,
)

baseline_f1 = f1_score(
    y_clf_test,
    y_pred_clf,
    zero_division=0,
)

baseline_auc = roc_auc_score(
    y_clf_test,
    y_proba_clf,
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_clf_test, y_pred_clf))

print("\nClassification Report:")
print(
    classification_report(
        y_clf_test,
        y_pred_clf,
        zero_division=0,
    )
)

print("Accuracy:", baseline_accuracy)
print("Precision:", baseline_precision)
print("Recall:", baseline_recall)
print("F1 Score:", baseline_f1)
print("AUC:", baseline_auc)


# ROC curve
fpr, tpr, thresholds = roc_curve(
    y_clf_test,
    y_proba_clf,
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression AUC = {baseline_auc:.4f}",
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()

plt.text(
    0.60,
    0.20,
    f"AUC = {baseline_auc:.4f}",
)

plt.tight_layout()
plt.savefig(
    "part2/images/roc_curve.png",
    dpi=200,
)
plt.close()


# Decision threshold sensitivity
print("\n===== DECISION THRESHOLD SENSITIVITY =====")

threshold_results = []

for threshold in np.arange(0.30, 0.71, 0.10):
    threshold_predictions = (
        y_proba_clf >= threshold
    ).astype(int)

    precision = precision_score(
        y_clf_test,
        threshold_predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_clf_test,
        threshold_predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_clf_test,
        threshold_predictions,
        zero_division=0,
    )

    threshold_results.append({
        "Threshold": round(threshold, 2),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    })

threshold_table = pd.DataFrame(threshold_results)

print(threshold_table.to_string(index=False))

best_threshold_row = threshold_table.loc[
    threshold_table["F1"].idxmax()
]

print("\nThreshold with maximum F1:")
print(best_threshold_row)


# Strong regularization experiment
print("\n===== LOGISTIC REGRESSION C=0.01 =====")

regularized_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    C=0.01,
)

regularized_model.fit(
    X_train_scaled,
    y_clf_train,
)

regularized_predictions = regularized_model.predict(
    X_test_scaled
)

regularized_probabilities = regularized_model.predict_proba(
    X_test_scaled
)[:, 1]

regularized_precision = precision_score(
    y_clf_test,
    regularized_predictions,
    zero_division=0,
)

regularized_recall = recall_score(
    y_clf_test,
    regularized_predictions,
    zero_division=0,
)

regularized_auc = roc_auc_score(
    y_clf_test,
    regularized_probabilities,
)

regularization_comparison = pd.DataFrame({
    "Model": [
        "Logistic C=1.0",
        "Logistic C=0.01",
    ],
    "Precision": [
        baseline_precision,
        regularized_precision,
    ],
    "Recall": [
        baseline_recall,
        regularized_recall,
    ],
    "AUC": [
        baseline_auc,
        regularized_auc,
    ],
})

print("\nRegularization comparison:")
print(
    regularization_comparison.to_string(index=False)
)


# Bootstrap confidence interval for AUC difference
print("\n===== BOOTSTRAP AUC DIFFERENCE =====")

np.random.seed(42)

auc_differences = []

y_test_array = y_clf_test.to_numpy()

for _ in range(500):
    sample_indices = np.random.choice(
        len(y_test_array),
        size=len(y_test_array),
        replace=True,
    )

    sampled_y = y_test_array[sample_indices]

    baseline_sample_proba = y_proba_clf[sample_indices]

    regularized_sample_proba = (
        regularized_probabilities[sample_indices]
    )

    # AUC needs both classes in the sample
    if len(np.unique(sampled_y)) < 2:
        continue

    auc_c1 = roc_auc_score(
        sampled_y,
        baseline_sample_proba,
    )

    auc_c001 = roc_auc_score(
        sampled_y,
        regularized_sample_proba,
    )

    auc_differences.append(
        auc_c1 - auc_c001
    )


mean_auc_difference = np.mean(auc_differences)

lower_ci = np.percentile(
    auc_differences,
    2.5,
)

upper_ci = np.percentile(
    auc_differences,
    97.5,
)

print("Bootstrap samples used:", len(auc_differences))
print("Mean AUC difference:", mean_auc_difference)
print("2.5th percentile:", lower_ci)
print("97.5th percentile:", upper_ci)

if lower_ci > 0 or upper_ci < 0:
    print("The 95% confidence interval excludes zero.")
else:
    print("The 95% confidence interval includes zero.")


print("\nPart 2 completed successfully.")
print("ROC curve saved to part2/images/roc_curve.png")