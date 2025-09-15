import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the training dataset
df = pd.read_csv(r"C:\Users\Tze Huo Gucci Ho\Desktop\Git Projects\P7_UTRY_Py\Mendeley Datasets\Body Measurements _ original_CSV.csv")

# Fix column names (remove trailing spaces)
df.columns = df.columns.str.strip()

# Basic data exploration on the training dataset
print("\nHead\n", df.head())
print("\nColumns\n", df.columns)
print("\nDescribe\n", df.describe())
print("\nIsnull\n", df.isnull().sum())   # Gender has 1 isnull value

# Data Cleaning: Impute missing values. If categorical, use mode to impute by most frequent value; if numerical, impute by mean
def impute_missing_values(df):
    for column in df.columns:
        if df[column].isnull().sum() > 0:
            if df[column].dtype == 'object':
                mode_value = df[column].mode()[0]
                df[column] = df[column].fillna(mode_value)
            else:
                mean_value = df[column].mean()
                df[column] = df[column].fillna(mean_value)
    return df

# Make a copy of the original data, then clean the data by imputing missing values in the copy
cleaned_df = impute_missing_values(df.copy())

print("\nIsnull\n", cleaned_df.isnull().sum())   # Gender now has 0 isnull values

# Data Visualization: Visualize the distribution of numerical features

# Function to convert inches to centimeters for specified columns
def convert_inches_to_cm(df, columns):
    df_cm = df.copy()
    for col in columns:
        df_cm[col] = df_cm[col] * 2.54
    return df_cm

numeric_cols = cleaned_df.select_dtypes(include=np.number).columns.tolist()

df_cm = convert_inches_to_cm(cleaned_df, numeric_cols)

# Function to plot histograms for numerical columns
def plot_histogram(df, numeric_cols):
    n_cols = 3
    n_rows = int(np.ceil(len(numeric_cols) / n_cols))   # Calculate number of rows needed based on number of columns
    plt.figure(figsize=(5 * n_cols, 4 * n_rows))   # Adjust figure size based on number of rows and columns

    for i, col in enumerate(numeric_cols):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.histplot(df[col], bins = 15, kde=True, color='skyblue')
        plt.title(f'Distribution of {col}')
        plt.xlabel(f'{col} (cm)')
        plt.ylabel('Count')

    plt.tight_layout()
    plt.show()

plot_histogram(df_cm, numeric_cols)

    # Observed outliers in:
    # Belly, outlier, 550+ cm
    # Waist, outlier, 250+ cm
    # ArmLength, outlier, 170+ cm

# Function to plot boxplots for numerical columns
def plot_boxplots(df, numeric_cols):
    n_cols = 1  # Stack boxplots vertically
    n_rows = len(numeric_cols)
    plt.figure(figsize=(10, 4 * n_rows))  # Taller figure for more features

    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.boxplot(x=df[col], color='lightgreen', orient='h')
        plt.title(f'Boxplot of {col}')
        plt.xlabel(f'{col} (cm)')

    plt.tight_layout()
    plt.show()

    # Outlier detection is performed within age groups
    # Each measurement is assessed relative to typical values for that age, not across all ages.

