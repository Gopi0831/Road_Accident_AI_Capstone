import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score



# Load the dataset
df = pd.read_csv("datasets/road safety accident dataset1.csv")

# Select the input columns
features = [
    "number_of_vehicles",
    "number_of_casualties",
    "day_of_week",
    "road_type",
    "speed_limit",
    "junction_detail",
    "light_conditions",
    "weather_conditions",
    "road_surface_conditions",
    "urban_or_rural_area"
]

X = df[features]
y = df["collision_severity"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)

# Create the baseline model
model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
train_prediction = model.predict(X_train)
test_prediction = model.predict(X_test)

# Calculate accuracy
train_accuracy = accuracy_score(y_train, train_prediction)
test_accuracy = accuracy_score(y_test, test_prediction)

print("\n===== DECISION TREE BASELINE =====")

print("\nTrain Accuracy:")
print(round(train_accuracy, 4))

print("\nTest Accuracy:")
print(round(test_accuracy, 4))

print("\nClassification Report:")
print(classification_report(y_test, test_prediction))

print("\nConfusion Matrix:")
print(classification_report(y_test, test_prediction, zero_division=0))

# Random Forest ensemble model
random_forest = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

random_forest.fit(X_train, y_train)

rf_train_prediction = random_forest.predict(X_train)
rf_test_prediction = random_forest.predict(X_test)

rf_train_accuracy = accuracy_score(y_train, rf_train_prediction)
rf_test_accuracy = accuracy_score(y_test, rf_test_prediction)

print("\n===== RANDOM FOREST MODEL =====")

print("\nTrain Accuracy:")
print(round(rf_train_accuracy, 4))

print("\nTest Accuracy:")
print(round(rf_test_accuracy, 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        rf_test_prediction,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_test_prediction))

# Tune the Random Forest model
parameters = {
    "n_estimators": [100, 150],
    "max_depth": [10, 15],
    "min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    RandomForestClassifier(
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),
    parameters,
    cv=3,
    scoring="f1_macro",
    n_jobs=-1
)

print("\nTuning the Random Forest model...")

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

tuned_prediction = best_model.predict(X_test)

tuned_accuracy = accuracy_score(y_test, tuned_prediction)
tuned_f1 = f1_score(
    y_test,
    tuned_prediction,
    average="macro"
)

print("\n===== TUNED RANDOM FOREST =====")

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nTest Accuracy:")
print(round(tuned_accuracy, 4))

print("\nMacro F1 Score:")
print(round(tuned_f1, 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        tuned_prediction,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, tuned_prediction))

# Save the best model
joblib.dump(best_model, "part2/accident_severity_model.pkl")

print("\nBest model saved successfully")