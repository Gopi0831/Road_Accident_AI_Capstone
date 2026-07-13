import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the road accident dataset
df = pd.read_csv("datasets/road safety accident dataset1.csv")

print("===== ROAD ACCIDENT DATA ANALYSIS =====")

# Check the dataset
print("\nDataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nStatistical summary:")
print(df.describe())

# Make a copy before cleaning
df_clean = df.copy()

# Fill missing numeric values with median
df_clean["location_easting_osgr"] = df_clean["location_easting_osgr"].fillna(
    df_clean["location_easting_osgr"].median()
)

df_clean["location_northing_osgr"] = df_clean["location_northing_osgr"].fillna(
    df_clean["location_northing_osgr"].median()
)

df_clean["longitude"] = df_clean["longitude"].fillna(
    df_clean["longitude"].median()
)

df_clean["latitude"] = df_clean["latitude"].fillna(
    df_clean["latitude"].median()
)

# Fill missing text values
df_clean["local_authority_highway_current"] = df_clean[
    "local_authority_highway_current"
].fillna("Unknown")

# Remove duplicate rows
df_clean = df_clean.drop_duplicates()

print("\nMissing values after cleaning:")
print(df_clean.isnull().sum().sum())

print("\nDuplicate rows after cleaning:")
print(df_clean.duplicated().sum())

# Convert severity codes into readable names
severity_names = {
    1: "Fatal",
    2: "Serious",
    3: "Slight"
}

df_clean["severity_name"] = df_clean["collision_severity"].map(severity_names)

# Convert day codes into readable names
day_names = {
    1: "Sunday",
    2: "Monday",
    3: "Tuesday",
    4: "Wednesday",
    5: "Thursday",
    6: "Friday",
    7: "Saturday"
}

df_clean["day_name"] = df_clean["day_of_week"].map(day_names)

# Create folder for plots
os.makedirs("part1/images", exist_ok=True)

# Plot 1: Collision severity
plt.figure(figsize=(8, 5))
sns.countplot(data=df_clean, x="severity_name")
plt.title("Distribution of Collision Severity")
plt.xlabel("Collision Severity")
plt.ylabel("Number of Collisions")
plt.tight_layout()
plt.savefig("part1/images/collision_severity_distribution.png")
plt.close()

# Plot 2: Collisions by day
plt.figure(figsize=(10, 5))
sns.countplot(data=df_clean, x="day_name")
plt.title("Road Collisions by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Number of Collisions")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("part1/images/collisions_by_day.png")
plt.close()

# Plot 3: Speed limit distribution
plt.figure(figsize=(8, 5))
plt.hist(df_clean["speed_limit"], bins=10)
plt.title("Distribution of Speed Limits")
plt.xlabel("Speed Limit")
plt.ylabel("Number of Collisions")
plt.tight_layout()
plt.savefig("part1/images/speed_limit_distribution.png")
plt.close()

# Plot 4: Speed limit and severity
plt.figure(figsize=(8, 5))
sns.boxplot(data=df_clean, x="severity_name", y="speed_limit")
plt.title("Speed Limit by Collision Severity")
plt.xlabel("Collision Severity")
plt.ylabel("Speed Limit")
plt.tight_layout()
plt.savefig("part1/images/speed_limit_by_severity.png")
plt.close()

# Plot 5: Correlation heatmap
columns = [
    "collision_severity",
    "number_of_vehicles",
    "number_of_casualties",
    "day_of_week",
    "speed_limit",
    "light_conditions",
    "weather_conditions",
    "road_surface_conditions"
]

correlation = df_clean[columns].corr()

plt.figure(figsize=(10, 7))
sns.heatmap(correlation, annot=True, fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("part1/images/correlation_heatmap.png")
plt.close()

# Print some findings
print("\nCollision severity count:")
print(df_clean["severity_name"].value_counts())

print("\nAverage number of casualties:")
print(round(df_clean["number_of_casualties"].mean(), 2))

print("\nAverage number of vehicles:")
print(round(df_clean["number_of_vehicles"].mean(), 2))

print("\nPart 1 completed successfully")