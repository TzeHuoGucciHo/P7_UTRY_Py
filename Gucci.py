
# ML task is multivariate imputation
# ChatGPT and Copilot assisted in formatting, structuring, and writing this code.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.spatial.distance import mahalanobis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler
from pandas.api.types import is_numeric_dtype, is_categorical_dtype



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
# 6. Data Splitting
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
# 7. Transform Skewed Features
# ---------------------------
def transform_features(train_df, val_df, test_df, numeric_cols):
    """
    Applies a Yeo-Johnson power transformation to reduce skewness.
    Fits only on the training set to avoid data leakage (no peeking).
    Then applies the fitted transformer to validation and test sets.
    """
    transformer = PowerTransformer(method='yeo-johnson', standardize=False)  # no scaling, just transform

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
# 8. Scale Features
# ---------------------------
def scale_features(train_df, val_df, test_df, numeric_cols):
    """
    Standardizes numerical features (mean=0, std=1).
    Fits only on the training set to avoid data leakage.
    Apply the fitted scaler to validation and test sets.
    """
    scaler = StandardScaler()

    # Fit on train, apply to all
    train_scaled = scaler.fit_transform(train_df[numeric_cols])
    val_scaled = scaler.transform(val_df[numeric_cols])
    test_scaled = scaler.transform(test_df[numeric_cols])

    # Replace numeric cols with scaled values
    train_df_scaled = train_df.copy()
    val_df_scaled = val_df.copy()
    test_df_scaled = test_df.copy()

    train_df_scaled[numeric_cols] = train_scaled
    val_df_scaled[numeric_cols] = val_scaled
    test_df_scaled[numeric_cols] = test_scaled

    return train_df_scaled, val_df_scaled, test_df_scaled



# ---------------------------
# Main Workflow
# ---------------------------
def main():
    filepath = r"C:\Users\Tze Huo Gucci Ho\Desktop\Git Projects\P7_UTRY_Py\Mendeley Datasets\Body Measurements _ original_CSV.csv"
    df = load_and_clean_data(filepath)  # Load data and impute missing values

    # Identify categorical and numerical columns
    categorical_cols = ['Gender']   # explicitly mark Gender as categorical
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    # Exclude categorical and non-length features from length-based conversion
    length_cols = [col for col in numeric_cols if col not in ['Age'] + categorical_cols]

    # Convert length columns from inches to cm for interpretability
    df_cm = convert_inches_to_cm(df, length_cols)

    # Initial data overview
    print("\nHead:\n", df_cm.head())
    print("\nDescribe:\n", df_cm.describe())
    print("\nColumns\n", df.columns)
    print("\nMissing values:\n", df_cm.isnull().sum())

    # Visualizations to observe skewness, outliers, and distribution shape
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

    print(correlation_analysis(df_clean, numeric_cols, threshold=0.9)) # No highly correlated (0.9) features but there are some moderate correlations (0.5-0.8)

    # Now, we split the dataset before transforming and scaling, to avoid data leakage.
    # We split the dataset by rows, not columns, as we want to predict all body measurements per individual person.
    # Because every measurement can be both a predictor and a target, depending on the context.
    # We do not want to predict only one measurement, as that would be a univariate imputation.
    # We want to predict multiple measurements simultaneously, hence multivariate imputation.

    # Split dataset into train(70%)/val(15%)/test(15%) (by rows, i.e. people)
    train_df, val_df, test_df = split_dataset(df_clean, test_size=0.15, val_size=0.176)
    # Plot train set
    plot_plots(train_df, numeric_cols, length_cols) # We observed a gap in headcircumference but after checking, it might be a gap in the data emphasized by the transformation. Might not affect our ML.
    # sns.histplot(df_clean["HeadCircumference"], bins=20, kde=True)
    # plt.show()

    # Transform skewed features
    train_df, val_df, test_df = transform_features(train_df, val_df, test_df, numeric_cols)
    # Plot train set after transformation
    plot_plots(train_df, numeric_cols, length_cols)

    # Scale features
    train_df, val_df, test_df = scale_features(train_df, val_df, test_df, numeric_cols)
    # Plot train set after scaling
    plot_plots(train_df, numeric_cols, length_cols)

if __name__ == "__main__":
    main()
