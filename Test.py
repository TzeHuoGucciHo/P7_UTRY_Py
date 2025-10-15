import time
import math
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
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error



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

def plot_skewness_comparison(before_skew, after_skew):
    """
    Plots a horizontal bar chart comparing skewness before and after transformation.
    """
    skew_df = pd.DataFrame({
        'Before': before_skew,
        'After': after_skew
    })

    skew_df.plot(kind='barh', figsize=(10, 6), color=['salmon', 'skyblue'])
    plt.title('Feature Skewness Before vs After Transformation')
    plt.xlabel('Skewness')
    plt.axvline(0, color='gray', linestyle='--', linewidth=1)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

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
    transformer = PowerTransformer(method='yeo-johnson', standardize=True)  # standardize=True applies StandardScaler after transformation

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
    We could use RobustScaler to reduce the influence of extreme values when computing Mahalanobis distances,
    making outlier detection less sensitive to a few unusually large or small measurements.
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
# Data Preparation (Simulate Missing Data)
# ---------------------------
def simulate_missing_data(df, missing_rate=0.2, random_state=42, prefix=""):
    """
    Randomly masks a percentage of data as missing (NaN) to simulate real cases.
    Returns a dataframe with masked values and a mask indicating which values were masked.
    """
    np.random.seed(random_state)
    df_masked = df.copy()
    mask = pd.DataFrame(False, index=df.index, columns=df.columns)

    # Select only numeric columns to mask
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in ['Gender']]
    total_cells = len(df) * len(numeric_cols)
    total_to_mask = int(total_cells * missing_rate)

    # Select a random sample (row, col) positions to mask
    row_indices = np.random.choice(df.index, size=total_to_mask, replace=True)
    col_indices = np.random.choice(numeric_cols, size=total_to_mask, replace=True)

    # For each selected position, set to NaN and update mask
    for row, col in zip(row_indices, col_indices):
        df_masked.at[row, col] = np.nan
        mask.at[row, col] = True

    print(f"{prefix}: Percentage of missing values simulated: {missing_rate * 100:.0f}% (excluding Gender column)")

    return df_masked, mask

# ---------------------------
# Iterative Imputer + Performance Measurement
# ---------------------------
def iterative_imputer_once(
        estimator,
        train_df,
        target_masked_df,
        numeric_cols,
        reference_full_df=None,
        sample_posterior=False,
        max_iter=10,
        tol=1e-3,
        random_state=42,
        verbose=0
):
    """

    """
    train_X = train_df[numeric_cols].copy() # Get numeric columns from training set (full, no missing)
    target_X = target_masked_df[numeric_cols].copy()    # Get numeric columns from target set (val/test with missing values)

    # Define the IterativeImputer
    imputer = IterativeImputer(
        estimator=estimator,
        sample_posterior=sample_posterior,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        verbose=verbose
    )

    start = time.perf_counter() # Start recording runtime
    imputer.fit(train_X.values) # Fit only on training set to avoid data leakage
    imputed_array = imputer.transform(target_X.values)  # Apply imputer to target set
    elapsed = time.perf_counter() - start   # Calculate elapsed runtime

    imputed_df = target_masked_df.copy()    # Copy the target masked dataframe
    imputed_df[numeric_cols] = imputed_array    # Replace numeric columns with imputed values in the copy

    metrics = {"time_sec": elapsed}
    masked_locs = target_X.isnull() # Locations that were originally masked in the target set
    n_masked = int(masked_locs.values.sum())    # Count of number of masked values

    true_vals = reference_full_df[numeric_cols].values[masked_locs.values]  # True values from the reference full dataframe at the masked locations
    pred_vals = imputed_array[masked_locs.values]   # Predicted (imputed) values at the masked locations

    # Compute MAE, MSE, and RMSE only on the originally masked locations, comparing to true values against the predicted values
    mae = mean_absolute_error(true_vals, pred_vals)
    mse = mean_squared_error(true_vals, pred_vals)
    rmse = math.sqrt(mse)

    # Add metrics to dictionary to return
    metrics.update({"mae": mae, "rmse": rmse, "n_masked": n_masked})

    return imputed_df, imputer, metrics

def evaluate_imputer_candidates(
        candidates,
        train_df,
        target_masked_df,
        numeric_cols,
        reference_full_df=None,
        sample_posterior=False,
        max_iter=10,
        tol=1e-3,
        random_state=42,
        verbose=0,
        save_csv=False,
        plot_results=False
):
    """

    """
    results = []
    imputers = {}
    imputed_dfs = {}

    print("\n--- Running Imputer Candidates ---\n")
    for name, est in candidates.items():    # For each candidate estimator
        print(f">>> Running {name} ...")
        try:
            imputed_df, imputer, metrics = iterative_imputer_once(
                estimator=est,
                train_df=train_df,
                target_masked_df=target_masked_df,
                numeric_cols=numeric_cols,
                reference_full_df=reference_full_df,
                sample_posterior=sample_posterior,
                max_iter=max_iter,
                tol=tol,
                random_state=random_state,
                verbose=verbose
            )

            # Set up metrics and results to print
            metrics["model"] = name
            results.append(metrics)
            imputers[name] = imputer
            imputed_dfs[name] = imputed_df
            print(f"{name} done in {metrics['time_sec']:.2f}s | MAE={metrics.get('mae', np.nan):.4f} | RMSE={metrics.get('rmse', np.nan):.4f}")

            # Sanity check to ensure no NaNs remain in imputed dataframe
        except Exception as e:
            print(f"{name} failed: {e}")
            results.append({
                "model": name, "mae": np.nan, "rmse": np.nan,
                "n_masked": 0, "time_sec": np.nan, "error": str(e)
            })
            imputers[name] = None
            imputed_dfs[name] = None

    # Compile results into a dataframe and sort by RMSE (if available) and print summary comparing model performance
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=["rmse", "mae"], ascending=True)
    results_df.reset_index(drop=True, inplace=True)
    print("\n--- Imputer Performance Summary ---\n")
    print(results_df[["model", "mae", "rmse", "n_masked", "time_sec"]])

    # For saving results to CSV
    if save_csv:
        results_df.to_csv("imputer_results.csv", index=False)
        print("Saved results to imputer_results.csv")

    # For plotting results comparison
    if plot_results and not results_df.empty:
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        results_df.plot.bar(x='model', y='mae', ax=ax[0], legend=False, title='MAE')
        results_df.plot.bar(x='model', y='rmse', ax=ax[1], legend=False, title='RMSE')
        results_df.plot.bar(x='model', y='time_sec', ax=ax[2], legend=False, title='Time (s)')
        for a in ax:
            a.set_xlabel('')
            a.set_xticklabels(results_df['model'], rotation=30, ha='right')
        plt.tight_layout()
        plt.show()

    return results_df, imputers, imputed_dfs



# ---------------------------
# Main Workflow
# ---------------------------
def main():
    correlation_threshold = 0.9  # The threshold is 90% for considering features as highly correlated
    filepath = r"C:\Users\Tze Huo Gucci Ho\Desktop\Git Projects\P7_UTRY_Py\Mendeley Datasets\Body Measurements _ original_CSV.csv"
    df = load_and_clean_data(filepath)

    # Data overview
    data_overview(df)

    df = df.drop_duplicates()

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
    #data_overview(train_df)

    # Ordinal/label encoding, maps 1 to 0 and 2 to 1. Replaces the existing gender column with the new binary values.
    mapping = {1: 0, 2: 1}
    train_df['Gender'] = train_df['Gender'].map(mapping)
    val_df['Gender'] = val_df['Gender'].map(mapping)
    test_df['Gender'] = test_df['Gender'].map(mapping)

    numeric_cols = train_df.select_dtypes(include=np.number).columns.tolist()   # Columns for numerical features, incl. Age and gender encoded
    num_cols_no_gender = [col for col in numeric_cols if col not in ['Gender']]  # Exclude Gender column from transformation and scaling
    length_cols = [col for col in numeric_cols if col not in ['Age', 'Gender']] # Columns for specifically body measurements

    #train_df_cm = convert_inches_to_cm(train_df, length_cols)   # Converts inches to cm for body measurements

    # Data overview
    #plot_plots(train_df, num_cols_no_gender, length_cols)
    #shapiro_wilk_test(train_df, num_cols_no_gender)
    #correlation_analysis(train_df, num_cols_no_gender, correlation_threshold)

    #print("Before transformation:\n", train_df[num_cols_no_gender].skew().sort_values(ascending=False))
    train_df_trans, val_df_trans, test_df_trans = transform_and_scale(train_df, val_df, test_df, num_cols_no_gender)
    #print("\nAfter transformation:\n", train_df_trans[num_cols_no_gender].skew().sort_values(ascending=False))
    # Skewness was measured before and after transformation.
    # Features like Belly, HeadCircumference, and ShoulderWidth were highly right-skewed (>5),
    # but Yeo-Johnson transformation reduced these to near-symmetric distributions (~0),
    # improving normality assumptions for later modeling.

    #before_skew = train_df[num_cols_no_gender].skew().sort_values(ascending=False)
    #train_df_trans, val_df_trans, test_df_trans = transform_and_scale(train_df, val_df, test_df, num_cols_no_gender)
    #after_skew = train_df_trans[num_cols_no_gender].skew().sort_values(ascending=False)
    #plot_skewness_comparison(before_skew, after_skew)

    # Data overview
    #plot_plots(train_df_trans, num_cols_no_gender, length_cols)
    #shapiro_wilk_test(train_df_trans, num_cols_no_gender)
    # Although the Shapiro–Wilk test rejects strict normality for several features,
    # visual inspection of histograms and Q-Q plots indicates that most features are approximately normally distributed.
    # Minor skewness and tail deviations remain but are acceptable for multivariate analysis assumptions.

    #correlation_analysis(train_df_trans, num_cols_no_gender, correlation_threshold)

    train_df_cleaned = mahalanobis_chi_outliers(train_df_trans, length_cols, alpha=0.0001, remove=True)

    # Data overview after outlier removal
    #plot_plots(train_df_cleaned, num_cols_no_gender, length_cols)   # Some features show multimodal distributions after outlier removal, likely due population subgroups (e.g., Gender, Age)
    #shapiro_wilk_test(train_df_cleaned, num_cols_no_gender)
    #correlation_analysis(train_df_cleaned, num_cols_no_gender, correlation_threshold)

    train_full = train_df_cleaned

    val_full = val_df_trans
    val_masked_df, val_mask = simulate_missing_data(val_full, missing_rate=0.2, random_state=42, prefix="val")  # Simulate 20% missing data

    test_full = test_df_trans
    test_masked_df, test_mask = simulate_missing_data(test_full, missing_rate=0.2, random_state=42, prefix="test")  # Simulate 20% missing data

    # Candidate estimators to embed inside IterativeImputer
    candidates = {
        "BayesianRidge": BayesianRidge(),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "RandomForest": RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    }

    # Evaluate candidates on validation masked set
    results_df, imputers, imputed_dfs = evaluate_imputer_candidates(
        candidates=candidates,
        train_df=train_full,
        target_masked_df=val_masked_df,
        numeric_cols=num_cols_no_gender,
        reference_full_df=val_full,
        sample_posterior=False,
        max_iter=10,
        random_state=42,
        verbose=0,
        save_csv=True,
        plot_results=True
    )

    print("\nBest candidate(s) by RMSE:\n", results_df.sort_values('rmse').head())

if __name__ == "__main__":
    main()