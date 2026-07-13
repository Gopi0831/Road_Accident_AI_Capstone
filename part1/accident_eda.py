import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Create the output folder for plots
os.makedirs("part1/images", exist_ok=True)


# Load the road accident dataset
df = pd.read_csv("datasets/road safety accident dataset1.csv")

print("===== DATASET OVERVIEW =====")

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn data types:")
print(df.dtypes)

print("\nDataset shape:")
print(df.shape)


# Null value analysis
print("\n===== NULL VALUE ANALYSIS =====")

null_count = df.isnull().sum()
null_percentage = (df.isnull().sum() / df.shape[0]) * 100

null_table = pd.DataFrame({
    "Null Count": null_count,
    "Null Percentage": null_percentage
})

print(null_table)

columns_above_20 = null_percentage[null_percentage > 20]

print("\nColumns with more than 20% missing values:")

if len(columns_above_20) == 0:
    print("No columns have more than 20% missing values.")
else:
    print(columns_above_20)


# Store null percentages before duplicate removal
null_percentage_before = null_percentage.copy()


# Duplicate detection and removal
print("\n===== DUPLICATE ANALYSIS =====")

duplicate_count = df.duplicated().sum()
rows_before = df.shape[0]

print("Duplicate rows found:", duplicate_count)

df = df.drop_duplicates()

rows_after = df.shape[0]
rows_removed = rows_before - rows_after

print("Rows removed:", rows_removed)

null_percentage_after = (df.isnull().sum() / df.shape[0]) * 100

null_change = pd.DataFrame({
    "Before Duplicate Removal": null_percentage_before,
    "After Duplicate Removal": null_percentage_after
})

print("\nNull percentages before and after duplicate removal:")
print(null_change)

if null_percentage_before.equals(null_percentage_after):
    print("\nDuplicate removal did not change null percentages.")
else:
    print("\nSome null percentages changed after duplicate removal.")


# Impute numeric columns below 20% missing values
print("\n===== NUMERIC NULL IMPUTATION =====")

numeric_columns = df.select_dtypes(include=np.number).columns

for col in numeric_columns:
    missing_percentage = (df[col].isnull().sum() / df.shape[0]) * 100

    if 0 < missing_percentage < 20:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)

        print(
            col,
            "- filled missing values with median:",
            median_value
        )


# Data type correction and memory comparison
print("\n===== DATA TYPE CORRECTION =====")

memory_before = df.memory_usage(deep=True).sum()

print("Memory usage before conversion:", memory_before, "bytes")

print("Date dtype before conversion:", df["date"].dtype)

df["date"] = pd.to_datetime(
    df["date"],
    dayfirst=True,
    errors="coerce"
)

print("Date dtype after conversion:", df["date"].dtype)

print(
    "Local authority highway dtype before:",
    df["local_authority_highway"].dtype
)

df["local_authority_highway"] = (
    df["local_authority_highway"].astype("category")
)

print(
    "Local authority highway dtype after:",
    df["local_authority_highway"].dtype
)

memory_after = df.memory_usage(deep=True).sum()

print("Memory usage after conversion:", memory_after, "bytes")
print("Memory saved:", memory_before - memory_after, "bytes")


# Descriptive statistics
print("\n===== DESCRIPTIVE STATISTICS =====")

numeric_columns = df.select_dtypes(include=np.number).columns

print(df[numeric_columns].describe())


# Skewness analysis
print("\n===== SKEWNESS ANALYSIS =====")

skewness = df[numeric_columns].skew()
skewness = skewness.sort_values(
    key=lambda values: values.abs(),
    ascending=False
)

print(skewness)

most_skewed_column = skewness.index[0]

print("\nMost skewed numeric column:", most_skewed_column)
print("Skewness value:", skewness.iloc[0])


# Mean and median comparison for two most skewed columns
print("\n===== IMPUTATION STRATEGY COMPARISON =====")

top_two_skewed = skewness.index[:2]

for col in top_two_skewed:
    column_mean = df[col].mean()
    column_median = df[col].median()

    print("\nColumn:", col)
    print("Skewness:", skewness[col])
    print("Mean:", column_mean)
    print("Median:", column_median)

    df[col] = df[col].fillna(column_median)

print("\nRemaining null values in the two most skewed columns:")
print(df[top_two_skewed].isnull().sum())


# IQR outlier analysis
print("\n===== IQR OUTLIER ANALYSIS =====")

outlier_columns = [
    "number_of_vehicles",
    "number_of_casualties"
]

for col in outlier_columns:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    outlier_count = (
        (df[col] < lower_bound) |
        (df[col] > upper_bound)
    ).sum()

    print("\nColumn:", col)
    print("Q1:", q1)
    print("Q3:", q3)
    print("IQR:", iqr)
    print("Lower bound:", lower_bound)
    print("Upper bound:", upper_bound)
    print("Outlier rows:", outlier_count)


# Grouped aggregation
print("\n===== GROUPED AGGREGATION =====")

grouped_result = (
    df.groupby(
        "collision_severity",
        observed=False
    )["number_of_casualties"]
    .agg(["mean", "std", "count"])
)

print(grouped_result)

highest_mean_group = grouped_result["mean"].idxmax()
highest_std_group = grouped_result["std"].idxmax()

highest_group_mean = grouped_result["mean"].max()
lowest_group_mean = grouped_result["mean"].min()

mean_ratio = highest_group_mean / lowest_group_mean

print("\nGroup with highest mean:", highest_mean_group)
print("Group with highest standard deviation:", highest_std_group)
print("Highest mean to lowest mean ratio:", mean_ratio)


# Pearson correlation
print("\n===== PEARSON CORRELATION =====")

pearson_corr = df[numeric_columns].corr()

print(pearson_corr)

pearson_pairs = pearson_corr.abs().where(
    np.triu(
        np.ones(pearson_corr.shape),
        k=1
    ).astype(bool)
)

highest_corr_pair = pearson_pairs.stack().idxmax()
highest_corr_value = pearson_corr.loc[
    highest_corr_pair[0],
    highest_corr_pair[1]
]

print("\nHighest absolute Pearson correlation pair:")
print(highest_corr_pair)
print("Pearson correlation:", highest_corr_value)


# Spearman correlation
print("\n===== SPEARMAN CORRELATION =====")

spearman_corr = df[numeric_columns].corr(method="spearman")

print(spearman_corr)


# Compare Pearson and Spearman correlations
print("\n===== PEARSON AND SPEARMAN DIFFERENCE =====")

difference_matrix = (spearman_corr - pearson_corr).abs()

print(difference_matrix)

correlation_comparison = []

for i in range(len(numeric_columns)):
    for j in range(i + 1, len(numeric_columns)):
        col1 = numeric_columns[i]
        col2 = numeric_columns[j]

        pearson_value = pearson_corr.loc[col1, col2]
        spearman_value = spearman_corr.loc[col1, col2]

        difference = abs(spearman_value - pearson_value)

        correlation_comparison.append({
            "Column 1": col1,
            "Column 2": col2,
            "Pearson": pearson_value,
            "Spearman": spearman_value,
            "Absolute Difference": difference
        })

difference_table = pd.DataFrame(correlation_comparison)

difference_table = difference_table.sort_values(
    "Absolute Difference",
    ascending=False
)

top_three_differences = difference_table.head(3)

print("\nTop 3 correlation differences:")
print(top_three_differences.to_string(index=False))


# Visualization 1 - Line plot
plt.figure(figsize=(10, 5))

sorted_casualties = df["number_of_casualties"].sort_index()

plt.plot(sorted_casualties)

plt.title("Number of Casualties Across Collision Records")
plt.xlabel("Row Index")
plt.ylabel("Number of Casualties")

plt.tight_layout()
plt.savefig(
    "part1/images/line_plot_casualties.png"
)
plt.close()


# Visualization 2 - Bar chart
severity_mean = (
    df.groupby(
        "collision_severity",
        observed=False
    )["number_of_casualties"]
    .mean()
)

plt.figure(figsize=(8, 5))

plt.bar(
    severity_mean.index.astype(str),
    severity_mean.values
)

plt.title("Mean Casualties by Collision Severity")
plt.xlabel("Collision Severity")
plt.ylabel("Mean Number of Casualties")

plt.tight_layout()
plt.savefig(
    "part1/images/bar_chart_mean_casualties.png"
)
plt.close()


# Visualization 3 - Histogram
plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x=most_skewed_column,
    bins=20
)

plt.title(
    f"Distribution of {most_skewed_column}"
)
plt.xlabel(most_skewed_column)
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig(
    "part1/images/histogram_most_skewed.png"
)
plt.close()


# Visualization 4 - Scatter plot
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="number_of_vehicles",
    y="number_of_casualties",
    alpha=0.5
)

plt.title("Number of Vehicles vs Number of Casualties")
plt.xlabel("Number of Vehicles")
plt.ylabel("Number of Casualties")

plt.tight_layout()
plt.savefig(
    "part1/images/scatter_vehicles_casualties.png"
)
plt.close()


# Visualization 5 - Box plot
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="collision_severity",
    y="speed_limit"
)

plt.title("Speed Limit by Collision Severity")
plt.xlabel("Collision Severity")
plt.ylabel("Speed Limit")

plt.tight_layout()
plt.savefig(
    "part1/images/boxplot_speed_severity.png"
)
plt.close()


# Correlation heatmap
plt.figure(figsize=(30, 24))

sns.heatmap(
    pearson_corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    annot_kws={"size": 6}
)

plt.title("Pearson Correlation Heatmap of Numeric Columns")

plt.tight_layout()
plt.savefig(
    "part1/images/correlation_heatmap.png",
    dpi=200
)
plt.close()


# Final null check
print("\n===== FINAL NULL VALUES =====")

print(df.isnull().sum())


# Save the cleaned dataset
output_file = "part1/cleaned_data.csv"

df.to_csv(output_file, index=False)

print("\nCleaned dataset saved successfully:")
print(output_file)

print("\nAll Part 1 visualizations saved in part1/images.")