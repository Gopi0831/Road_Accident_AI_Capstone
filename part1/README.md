# Part 1 - Road Accident Data Analysis

## Project Overview

This part of the project focuses on exploring and cleaning the road safety accident dataset. The main aim is to understand the data and identify some basic patterns before building a machine learning model.

## Dataset

The project uses the 2023 road safety collision dataset published as road safety open data by the UK Department for Transport.

The dataset contains 104,258 rows and 44 columns. It includes information about collision severity, number of vehicles, number of casualties, speed limits, road conditions, weather conditions and light conditions.

## Data Cleaning

I first checked the dataset shape, column information, missing values and duplicate rows.

A small number of missing values were found in the location columns. Missing numeric values were filled using the median of the respective column.

Missing values in `local_authority_highway_current` were filled with `Unknown`.

Duplicate rows were also checked and removed if present.

## Exploratory Data Analysis

The following visualizations were created:

1. Distribution of collision severity
2. Road collisions by day of the week
3. Distribution of speed limits
4. Speed limit by collision severity
5. Correlation heatmap of selected features

The plots are saved in the `images` folder.

## Key Findings

The dataset contains more slight collisions than serious and fatal collisions.

Road collisions occur throughout the week, with differences in the number of collisions recorded on different days.

The speed limit distribution shows that collisions are recorded across different road speed limits.

The box plot helps compare speed limits for fatal, serious and slight collisions.

The correlation heatmap shows the relationship between collision severity and selected numerical road accident features.

## How to Run

Install the required Python libraries:

```bash
pip install pandas matplotlib seaborn