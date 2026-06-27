# ==========================================
# DATA EXPLORATION USING IRIS DATASET
# Author: Divyansh Dodiyar
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris

# ------------------------------------------
# Load Dataset
# ------------------------------------------

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df['species'] = iris.target

species_names = {
    0: 'Setosa',
    1: 'Versicolor',
    2: 'Virginica'
}

df['species'] = df['species'].map(species_names)

print("\n========== FIRST 5 ROWS ==========\n")
print(df.head())

# ------------------------------------------
# Dataset Information
# ------------------------------------------

print("\n========== DATASET INFO ==========\n")
print(df.info())

print("\n========== SHAPE ==========\n")
print(df.shape)

print("\n========== COLUMN NAMES ==========\n")
print(df.columns)

# ------------------------------------------
# Missing Values
# ------------------------------------------

print("\n========== MISSING VALUES ==========\n")
print(df.isnull().sum())

# ------------------------------------------
# Statistical Summary
# ------------------------------------------

print("\n========== STATISTICAL SUMMARY ==========\n")
print(df.describe())

# ------------------------------------------
# Class Distribution
# ------------------------------------------

print("\n========== SPECIES COUNT ==========\n")
print(df['species'].value_counts())

# ==========================================
# VISUALIZATIONS
# ==========================================

# ------------------------------------------
# Histogram
# ------------------------------------------

df.hist(
    figsize=(10, 8),
    bins=20
)

plt.suptitle("Feature Distributions")
plt.tight_layout()

plt.savefig("histogram.png")
plt.show()

# ------------------------------------------
# Box Plot
# ------------------------------------------

plt.figure(figsize=(10,6))

sns.boxplot(data=df.iloc[:,0:4])

plt.title("Box Plot of Features")

plt.savefig("boxplot.png")
plt.show()

# ------------------------------------------
# Pair Plot
# ------------------------------------------

sns.pairplot(
    df,
    hue='species'
)

plt.savefig("pairplot.png")
plt.show()

# ------------------------------------------
# Correlation Heatmap
# ------------------------------------------

plt.figure(figsize=(8,6))

numeric_df = df.iloc[:,0:4]

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")

plt.savefig("heatmap.png")
plt.show()

# ------------------------------------------
# Scatter Plot
# ------------------------------------------

plt.figure(figsize=(8,6))

sns.scatterplot(
    data=df,
    x='sepal length (cm)',
    y='petal length (cm)',
    hue='species'
)

plt.title("Sepal Length vs Petal Length")

plt.savefig("scatterplot.png")
plt.show()

print("\nProject Completed Successfully!") 