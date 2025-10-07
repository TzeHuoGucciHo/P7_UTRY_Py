import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.spatial.distance import mahalanobis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler
from pandas.api.types import is_numeric_dtype, is_categorical_dtype
from scipy.stats import chi2
from sklearn.impute import SimpleImputer


# ML task is multivariate imputation
# ChatGPT and Copilot assisted in formatting, structuring, and writing this code.

# ---------------------------
# Data Overview and Cleaning
# ---------------------------
def load_and_clean_data(filepath):
    """
    Loads CSV, strips column names.
    Returns cleaned dataframe.
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip() # Strip whitespace from column names
    return df

def data_overview(df):
    """
    Prints basic info, missing values, duplicates.
    """

    print("Data Overview:")
    print(df.info())
    print("\nHead:\n", df.head())
    print("\nDescribe:\n", df.describe())
    print("\nColumns\n", df.columns)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nDuplicates:", df.duplicated().sum())

def convert_inches_to_cm(df, length_cols):
    """
    Converts numerical columns from inches to cm (1 inch = 2.54 cm).
    """
    df[length_cols] = df[length_cols] * 2.54
    return df

# ---------------------------
# Split Dataset
# ---------------------------
def split_dataset(df, test_size=0.15, val_size=0.176, random_state=42):
    """
    Splits the data into training (70%), validation (15%), and test sets (15%).
    Removes 15% from the data, leaving 85%.
    Then removes 17.6% from the remaining 85%, which is ≈15% of the original data.
    0.176 * 85% ≈ 15% of original dataset
    Does this randomly by rows (people), not columns (features).
    """
    # First split into train+val and test for each person (rows)
    train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    # Then split train_val into train and validation
    # val_size here is proportion of the train_val_df (not original df)
    # 0.176 * 85% ≈ 15% of the total dataset
    train_df, val_df = train_test_split(train_val_df, test_size=val_size, random_state=random_state)

    print(f"Train shape: {train_df.shape}")
    print(f"Validation shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")

    return train_df, val_df, test_df

# ---------------------------
# Visualizations
# ---------------------------
def plot_histograms(df, numeric_cols, length_cols, suffix=""):
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
        xlabel = f"{col} {suffix}" if col in length_cols else col
        plt.xlabel(xlabel)
        plt.ylabel("Count")
        plt.title(f"Distribution of {col}")

    plt.tight_layout()
    plt.show()

def plot_boxplots(df, numeric_cols, length_cols, suffix=""):
    """
    Plots boxplots for numerical features.
    Length columns are labeled in cm.
    """
    plt.figure(figsize=(10, 4 * len(numeric_cols)))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(len(numeric_cols), 1, i)
        sns.boxplot(x=df[col], color='lightgreen', orient='h')
        xlabel = f"{col} {suffix}" if col in length_cols else col
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

def plot_plots(df, numeric_cols, length_cols, suffix=""):
    """
    Plots histograms, boxplots, and Q-Q plots for numerical features.
    Length columns are labeled in cm.
    """
    plot_histograms(df, numeric_cols, length_cols, suffix)
    plot_boxplots(df, numeric_cols, length_cols, suffix)
    plot_qq_plots(df, numeric_cols)

# ---------------------------
# Data Analysis
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
# Transformer and Scaler
# ---------------------------
def transform_and_scale(train_df, val_df, test_df, numeric_cols):
    """
    Applies a Yeo-Johnson power transformation to reduce skewness.
    The standardize=True option also scales to mean=0, std=1 (standard scaler).
    Fits only on the training set to avoid data leakage (no peeking).
    Then applies the fitted transformer to validation and test sets.

    Note that PowerTransformer applies non-linear transformation, altering relationships between features.
    This will affect the correlation matrix.
    """
    transformer = PowerTransformer(method='yeo-johnson', standardize=True)  # no scaling, just transform

    # Fit on train, apply to all
    train_transformed = transformer.fit_transform(train_df[numeric_cols])
    val_transformed = transformer.transform(val_df[numeric_cols])
    test_transformed = transformer.transform(test_df[numeric_cols])

    # Replace numeric cols with transformed values
    train_df_trans = train_df.copy()
    val_df_trans = val_df.copy()
    test_df_trans = test_df.copy()

    train_df_trans[numeric_cols] = train_transformed
    val_df_trans[numeric_cols] = val_transformed
    test_df_trans[numeric_cols] = test_transformed

    return train_df_trans, val_df_trans, test_df_trans

# ---------------------------
# Outlier Handling
# ---------------------------
def mahalanobis_chi_outliers(df, numeric_cols, alpha=0.01, remove=False):
    """
    Detects multivariate outliers using Mahalanobis distance and Chi-squared distribution.
    Flag and remove points with squared Mahalanobis distance from the mean that is so extreme,
    that the probability of observing such a point under the assumption of multivariate normality is less than alpha.
    """
    X = df[numeric_cols].values
    cov_matrix = np.cov(X, rowvar=False)
    cov_inv = np.linalg.inv(cov_matrix)
    mean_vector = np.mean(X, axis=0)

    distances = np.array([mahalanobis(row, mean_vector, cov_inv) for row in X])
    squared_distances = distances**2

    chi_threshold = chi2.ppf(1 - alpha, df=len(numeric_cols))
    outlier_mask = squared_distances > chi_threshold

    if remove:
        df_clean = df.loc[~outlier_mask].copy()
        print(f"Removed {np.sum(outlier_mask)} outliers (chi2 method)")
        return df_clean
    else:
        df_flagged = df.copy()
        df_flagged['outlier'] = outlier_mask
        print(f"Flagged {np.sum(outlier_mask)} outliers (chi2 method)")
        return df_flagged



# ---------------------------
# Main Workflow
# ---------------------------
def main():
    filepath = r"C:\Users\Tze Huo Gucci Ho\Desktop\Git Projects\P7_UTRY_Py\Mendeley Datasets\Body Measurements _ original_CSV.csv"
    df = load_and_clean_data(filepath)

    # Data overview
    data_overview(df)

    # Split dataset into train(70%)/val(15%)/test(15%) (by rows, i.e. people)
    train_df, val_df, test_df = split_dataset(df, test_size=0.15, val_size=0.176)

    # Impute missing values, fit only on train to avoid data leakage then apply to val and test
    num_cols = train_df.select_dtypes(include=[np.number]).columns
    num_imputer = SimpleImputer(strategy='mean')
    train_df[num_cols] = num_imputer.fit_transform(train_df[num_cols])
    val_df[num_cols] = num_imputer.transform(val_df[num_cols])
    test_df[num_cols] = num_imputer.transform(test_df[num_cols])

    cat_cols = train_df.select_dtypes(exclude=[np.number]).columns
    if len(cat_cols) > 0:   # Error occured earlier. Failsafe to only impute categorical columns if they exist
        cat_imputer = SimpleImputer(strategy='most_frequent')
        train_df[cat_cols] = cat_imputer.fit_transform(train_df[cat_cols])
        val_df[cat_cols] = cat_imputer.transform(val_df[cat_cols])
        test_df[cat_cols] = cat_imputer.transform(test_df[cat_cols])

    # Data overview
    train_df = train_df.drop_duplicates()
    data_overview(train_df)

    # Ordinal/label encoding, maps 1 to 0 and 2 to 1. Replaces the existing gender column with the new binary values.
    mapping = {1: 0, 2: 1}
    train_df['Gender'] = train_df['Gender'].map(mapping)
    val_df['Gender'] = val_df['Gender'].map(mapping)
    test_df['Gender'] = test_df['Gender'].map(mapping)

    numeric_cols = train_df.select_dtypes(include=np.number).columns.tolist()   # Columns for numerical features, incl. Age and gender encoded
    num_cols_no_gender = [col for col in numeric_cols if col not in ['Gender']]  # Exclude Gender column from transformation and scaling
    length_cols = [col for col in numeric_cols if col not in ['Age', 'Gender']] # Columns for specifically body measurements

    # train_df_cm = convert_inches_to_cm(train_df, length_cols)   # Converts inches to cm for body measurements

    # Data overview
    plot_plots(train_df, num_cols_no_gender, length_cols)
    shapiro_wilk_test(train_df, num_cols_no_gender)
    correlation_analysis(train_df, num_cols_no_gender, threshold=0.9)

    train_df_trans, val_df_trans, test_df_trans = transform_and_scale(train_df, val_df, test_df, num_cols_no_gender)

    # Data overview
    plot_plots(train_df_trans, num_cols_no_gender, length_cols)
    shapiro_wilk_test(train_df_trans, num_cols_no_gender)
    """
    Although the Shapiro–Wilk test rejects strict normality for several features, 
    visual inspection of histograms and Q-Q plots indicates that most features are approximately normally distributed. 
    Minor skewness and tail deviations remain but are acceptable for multivariate analysis assumptions.
    """

    correlation_analysis(train_df_trans, num_cols_no_gender, threshold=0.9)

    train_df_cleaned = mahalanobis_chi_outliers(train_df_trans, length_cols, alpha=0.001, remove=True)

    # Data overview after outlier removal
    plot_plots(train_df_cleaned, num_cols_no_gender, length_cols)
    shapiro_wilk_test(train_df_cleaned, num_cols_no_gender)
    correlation_analysis(train_df_cleaned, num_cols_no_gender, threshold=0.9)

if __name__ == "__main__":
    main()