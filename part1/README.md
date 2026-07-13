# Part 1 - Road Accident Data Cleaning and Exploratory Analysis

## Project Overview

This part of the project focuses on cleaning and exploring a road safety collision dataset before building a machine learning model.

The dataset contains 104,258 collision records and 44 columns. It includes information about collision severity, number of vehicles, number of casualties, road type, speed limit, junction details, weather conditions, lighting conditions, road surface conditions and accident location.

The purpose of this analysis is to understand the quality and structure of the data, handle missing values, inspect outliers and study relationships between important variables.

## Dataset

The project uses a public road safety collision dataset.

Dataset file used:

`datasets/road safety accident dataset1.csv`

The dataset is loaded using `pandas.read_csv()`.

Initial dataset shape:

- Rows: 104,258
- Columns: 44

The first five rows, column data types and dataset shape are printed when the script is executed.

## How to Run

Run the script from the main project folder:

```bash
python part1/accident_eda.py