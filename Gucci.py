import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.spatial.distance import mahalanobis

# ChatGPT and Copilot assisted in formatting, structuring, and writing this code.

# ---------------------------
# 1. Data Loading & Cleaning
# ---------------------------
def load_and_clean_data(filepath):
    """
    Loads CSV, strips column names, and imputes missing values.
    Returns cleaned dataframe.
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    df = impute_missing_values(df)
    return df


def impute_missing_values(df):
    """
    Imputes missing values in the dataframe:
        Mean for numerical columns.
        Mode for categorical columns.
    """
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].mean())
    return df


def convert_inches_to_cm(df, columns):
    """
    Converts numerical columns from inches to cm (1 inch = 2.54 cm).
    """
    df_cm = df.copy()
    df_cm[columns] = df_cm[columns] * 2.54
    return df_cm


# ---------------------------
# 2. Visualizations
# ---------------------------
def plot_histograms(df, numeric_cols, length_cols=[]):
    """
    Plots a histogram over the numerical features in the dataframe.
    Length columns are labeled in cm.
    """
    n_cols = 3
    n_rows = int(np.ceil(len(numeric_cols) / n_cols))
    plt.figure(figsize=(5 * n_cols, 4 * n_rows))

    for i, col in enumerate(numeric_cols):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.histplot(df[col], bins=20, kde=True, color='skyblue')
        xlabel = f"{col} (cm)" if col in length_cols else col
        plt.xlabel(xlabel)
        plt.ylabel("Count")
        plt.title(f"Distribution of {col}")

    plt.tight_layout()
    plt.show()


def plot_boxplots(df, numeric_cols, length_cols=[]):
    """
    Plots boxplots for numerical features.
    Length columns are labeled in cm.
    """
    plt.figure(figsize=(10, 4 * len(numeric_cols)))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(len(numeric_cols), 1, i)
        sns.boxplot(x=df[col], color='lightgreen', orient='h')
        xlabel = f"{col} (cm)" if col in length_cols else col
        plt.xlabel(xlabel)
        plt.title(f"Boxplot of {col}")
    plt.tight_layout()
    plt.show()


def plot_qq_plots(df, numeric_cols):
    """
    Plots Q-Q plots for numerical features to assess normality.
    Includes reference line for standard normal distribution.
    """
    n_cols = 3
    n_rows = int(np.ceil(len(numeric_cols) / n_cols))
    plt.figure(figsize=(5 * n_cols, 4 * n_rows))

    for i, col in enumerate(numeric_cols):
        plt.subplot(n_rows, n_cols, i + 1)
        stats.probplot(df[col], dist="norm", plot=plt)
        plt.title(f"Q-Q Plot of {col}")

    plt.tight_layout()
    plt.show()

def plot_plots(df, numeric_cols, length_cols=[]):
    """
    Plots histograms, boxplots, and Q-Q plots for numerical features.
    Length columns are labeled in cm.
    """
    plot_histograms(df, numeric_cols, length_cols)
    plot_boxplots(df, numeric_cols, length_cols)
    plot_qq_plots(df, numeric_cols)

# ---------------------------
# 3. Normality Tests
# ---------------------------
def shapiro_wilk_test(df, numeric_cols):
    """
    Performs Shapiro-Wilk test to assess normality of numerical features.
    Prints results indicating whether each feature is normally distributed.
    """
    results = {}
    for col in numeric_cols:
        stat, p_value = stats.shapiro(df[col])
        results[col] = (stat, p_value)
        print(f"{col}:", "Normal" if p_value > 0.05 else "Not normal")


# ---------------------------
# 4. Correlation Analysis
# ---------------------------
def correlation_analysis(df, numeric_cols, threshold=0.9):
    """
    Computes correlation matrix, plots heatmap, and optionally returns
    list of highly correlated feature pairs.
    """
    corr_matrix = df[numeric_cols].corr()

    # Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()

    # Identify highly correlated pairs
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

    if high_corr:
        print("Highly correlated feature pairs (>|{:.1f}|):".format(threshold))
        for f1, f2, corr_val in high_corr:
            print(f"{f1} & {f2}: {corr_val:.2f}")
    else:
        print("No highly correlated features above threshold.")

    return high_corr



# ---------------------------
# 5. Outlier Detection
# ---------------------------
def mahalanobis_outliers(df, numeric_cols, threshold=3.0, remove=True):
    """
    Flags or removes extreme multivariate outliers using Mahalanobis distance.
    """
    X = df[numeric_cols].values
    cov_matrix = np.cov(X, rowvar=False)
    cov_inv = np.linalg.inv(cov_matrix)
    mean_vector = np.mean(X, axis=0)

    # Compute Mahalanobis distance for each row
    distances = np.array([mahalanobis(row, mean_vector, cov_inv) for row in X])

    # Determine threshold (distance > threshold*std)
    dist_threshold = np.mean(distances) + threshold * np.std(distances)
    outlier_mask = distances > dist_threshold

    if remove:
        df_clean = df.loc[~outlier_mask].copy()
        print(f"Removed {np.sum(outlier_mask)} extreme multivariate outliers")
        return df_clean
    else:
        df_flagged = df.copy()
        df_flagged['outlier'] = outlier_mask
        print(f"Flagged {np.sum(outlier_mask)} extreme multivariate outliers")
        return df_flagged



# ---------------------------
# Main Workflow
# ---------------------------
def main():
    filepath = r"C:\Users\Tze Huo Gucci Ho\Desktop\Git Projects\P7_UTRY_Py\Mendeley Datasets\Body Measurements _ original_CSV.csv"
    df = load_and_clean_data(filepath)

    # Identify numerical columns and length columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    length_cols = [col for col in numeric_cols if col.lower() != 'age']

    # Convert length columns from inches to cm for interpretability. Does not affect statistics.
    df_cm = convert_inches_to_cm(df, length_cols)

    # Initial data overview
    print("\nHead:\n", df_cm.head())
    print("\nDescribe:\n", df_cm.describe())
    print("\nColumns\n", df.columns)
    print("\nMissing values:\n", df_cm.isnull().sum())

    # Visualizations
    plot_plots(df_cm, numeric_cols, length_cols)    # Several extreme outliers observed, and skewness

    # Normality test
    shapiro_wilk_test(df_cm, numeric_cols)  # No features are normally distributed, sensitive to outliers

    # Correlation analysis
    print(correlation_analysis(df_cm, numeric_cols, threshold=0.9)) # No highly correlated (0.9) features but there are some moderate correlations (0.5-0.8)

    # Outlier detection and removal using Mahalanobis distance (considering correlation between features)
    df_clean = mahalanobis_outliers(df_cm, numeric_cols, threshold=2.5, remove=True)    # Removed 13 extreme multivariate outliers, after considering the covariance structure, unlike univariate outliers.
    # Replot histograms, boxplots, and Q-Q plots
    plot_plots(df_clean, numeric_cols, length_cols) # Outliers removed, but some skewness still present
    # Re-run Shapiro-Wilk
    shapiro_wilk_test(df_clean, numeric_cols)   # Still no features that are normally distributed, sensitive to outliers still



if __name__ == "__main__":
    main()
