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

# Data Visualization: Visualize the distribution of numerical features

# Function to convert inches to centimeters for specified columns
def convert_inches_to_cm(df, columns):
    df_cm = df.copy()
    for col in columns:
        df_cm[col] = df_cm[col] * 2.54
    return df_cm

# Function to plot histograms for numerical columns
def plot_histogram(df, numeric_cols, length_cols):
    n_cols = 3
    n_rows = int(np.ceil(len(numeric_cols) / n_cols))
    plt.figure(figsize=(5 * n_cols, 4 * n_rows))

    for i, col in enumerate(numeric_cols):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.histplot(df[col], bins=15, kde=True, color='skyblue')
        plt.title(f'Distribution of {col}')
        xlabel = f"{col} (cm)" if col in length_cols else col   # Use cm only for length-based columns
        plt.xlabel(xlabel)
        plt.ylabel("Count")

    plt.tight_layout()
    plt.show()

    # Observed outliers in:
    # Belly, outlier, 550+ cm
    # Waist, outlier, 250+ cm
    # ArmLength, outlier, 170+ cm

# Function to plot boxplots for numerical columns
def plot_boxplots_extended(df, numeric_cols, length_cols, group_col="AgeGroup"):
    n_cols = 2  # One column for global, one for grouped
    n_rows = len(numeric_cols)
    plt.figure(figsize=(12 * n_cols, 4 * n_rows))

    for i, col in enumerate(numeric_cols):
        # Global boxplot
        plt.subplot(n_rows, n_cols, i * n_cols + 1)
        sns.boxplot(x=df[col], color="lightgreen", orient="h")
        xlabel = f"{col} (cm)" if col in length_cols else col
        plt.xlabel(xlabel)
        plt.title(f"Global Boxplot of {col}")

        # Age-grouped boxplot
        plt.subplot(n_rows, n_cols, i * n_cols + 2)
        sns.boxplot(x=group_col, y=col, data=df, palette="Set2")    # hue=None instead of palette="Set2" for no color
        ylabel = f"{col} (cm)" if col in length_cols else col
        plt.ylabel(ylabel)
        plt.xlabel(group_col)
        plt.title(f"{col} by {group_col}")

    plt.tight_layout()
    plt.show()

# Function to cap outliers based on IQR within each age group
# Function to cap outliers based on IQR within each age group and report them
def cap_outliers_local_iqr_report(df, numeric_cols, group_col="AgeGroup"):
    df_capped = df.copy()
    numeric_cols_to_cap = [col for col in numeric_cols if col.lower() != 'age']

    for col in numeric_cols_to_cap:
        for group, group_data in df_capped.groupby(group_col):
            Q1 = group_data[col].quantile(0.25)
            Q3 = group_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            idx = group_data.index
            for i in idx:
                val = df_capped.loc[i, col]
                if val < lower_bound:
                    print(f"Capped {col}: {val} -> {lower_bound} (AgeGroup: {group})")
                    df_capped.loc[i, col] = lower_bound
                elif val > upper_bound:
                    print(f"Capped {col}: {val} -> {upper_bound} (AgeGroup: {group})")
                    df_capped.loc[i, col] = upper_bound

    return df_capped


# Make a copy of the original data, then clean the data by imputing missing values in the copy
cleaned_df = impute_missing_values(df.copy())
print("\nIsnull\n", cleaned_df.isnull().sum())   # Gender now has 0 isnull values

# Convert length-based measurements from inches to centimeters
numeric_cols = cleaned_df.select_dtypes(include=np.number).columns.tolist()
length_cols = [col for col in numeric_cols if col.lower() != 'age'] # Exclude 'Age' from length conversion
df_cm = convert_inches_to_cm(cleaned_df, length_cols)

# Outlier detection is performed within age groups
# Each measurement is assessed relative to typical values for that age, not across all ages.
# For this purpose, we define age groups. We also edit the function for box plots to accept age grouping column.

# Dynamically define age bins based on the dataset
min_age = df_cm["Age"].min()
max_age = df_cm["Age"].max()

# Define age bins and labels
bins = [min_age, 12, 19, 39, 59, max_age + 1]  # +1 to include the max age in the last bin
labels = ["Child", "Teen", "Adult", "Middle-aged", "Senior"]

# Assign AgeGroup dynamically
df_cm["AgeGroup"] = pd.cut(df_cm["Age"], bins=bins, labels=labels, right=True, include_lowest=True)

plot_histogram(df_cm, numeric_cols, length_cols)
plot_boxplots_extended(df_cm, numeric_cols, length_cols, group_col="AgeGroup")

# Cap outliers based on local IQR within each age group
# Preserves all samples but constrains extreme values relative to age group
# Exclude "Age" from capping as it is not a measurement

# Apply local IQR capping to the dataset
df_capped = cap_outliers_local_iqr_report(df_cm, numeric_cols, group_col="AgeGroup")

# Optional: Visualize capped data
plot_boxplots_extended(df_capped, numeric_cols, length_cols, group_col="AgeGroup")
plot_histogram(df_capped, numeric_cols, length_cols)





