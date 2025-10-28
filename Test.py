import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version)

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
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid



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
def split_dataset(df, train_frac=0.7, val_frac=0.15, test_frac=0.15, random_state=42):
    """
    Splits dataframe into train, validation, and test sets in 70/15/15 ratio.
    """
    train_df, temp_df = train_test_split(df, train_size=train_frac, random_state=random_state)
    val_size = val_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(temp_df, train_size=val_size, random_state=random_state)
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

    start_fit = time.perf_counter() # Start recording runtime for fitting
    imputer.fit(train_X.values) # Fit only on training set to avoid data leakage
    fit_time = time.perf_counter() - start_fit  # Calculate elapsed runtime for fitting

    start_transform = time.perf_counter()    # Start recording runtime for transforming
    imputed_array = imputer.transform(target_X.values)  # Apply imputer to target set
    transform_time = time.perf_counter() - start_transform  # Calculate elapsed runtime for transforming

    elapsed = fit_time + transform_time

    imputed_df = target_masked_df.copy()    # Copy the target masked dataframe
    imputed_df[numeric_cols] = imputed_array    # Replace numeric columns with imputed values in the copy

    metrics = {
        "fit_time": fit_time,
        "transform_time": transform_time,
        "total_time": elapsed
    }

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
        verbose=0
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
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values(by=["rmse", "mae"], ascending=True)

            imputers[name] = imputer
            imputed_dfs[name] = imputed_df
            print(
                f"{name} done in {metrics['total_time']:.2f}s | MAE={metrics.get('mae', np.nan):.4f} | RMSE={metrics.get('rmse', np.nan):.4f}")

            # Sanity check to ensure no NaNs remain in imputed dataframe
        except Exception as e:
            print(f"{name} failed: {e}")
            results.append({
                "model": name, "mae": np.nan, "rmse": np.nan,
                "n_masked": 0, "total_time": np.nan, "error": str(e)
            })

            imputers[name] = None
            imputed_dfs[name] = None

    # Compile results into a dataframe and sort by RMSE (if available) and print summary comparing model performance
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=["rmse", "mae"], ascending=True)
    results_df.reset_index(drop=True, inplace=True)
    print("\n--- Imputer Performance Summary ---\n")
    print(results_df[["model", "mae", "rmse", "n_masked", "total_time"]])

    return results_df, imputers, imputed_dfs

# ---------------------------
# K-Fold Cross-Validation
# ---------------------------
def cross_validate_imputers(
    candidates,
    full_train_df,
    numeric_cols,
    n_splits=5,
    missing_rate=0.2,
    random_state=42
):
    """
    Runs k-fold cross-validation for imputer model comparison.
    Each fold trains on (k-1)/k of the data and tests on the remaining 1/k,
    simulating missing values each time.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    all_fold_results = []

    print(f"\n--- Starting {n_splits}-Fold Cross-Validation for Imputers ---\n")

    fold_num = 1
    for train_idx, val_idx in kf.split(full_train_df):
        print(f"\n=== Fold {fold_num}/{n_splits} ===")
        fold_num += 1

        fold_train = full_train_df.iloc[train_idx]
        fold_val = full_train_df.iloc[val_idx]

        # Simulate missing data on validation fold
        fold_val_masked, _ = simulate_missing_data(
            fold_val,
            missing_rate=missing_rate,
            random_state=random_state,
            prefix=f"Fold {fold_num}"
        )

        # Evaluate imputers using existing function
        results_df, _, _ = evaluate_imputer_candidates(
            candidates=candidates,
            train_df=fold_train,
            target_masked_df=fold_val_masked,
            numeric_cols=numeric_cols,
            reference_full_df=fold_val,
            sample_posterior=False,
            max_iter=10,
            random_state=random_state,
            verbose=0,
        )

        # Add fold index to results
        results_df["fold"] = fold_num
        all_fold_results.append(results_df)

    # Combine results from all folds
    combined_df = pd.concat(all_fold_results)

    # Average across folds per model
    summary_df = combined_df.groupby("model")[["mae", "rmse", "total_time"]].mean().reset_index()

    print("\n--- Cross-Validation Summary (Averaged over folds) ---\n")
    print(summary_df)

    return summary_df

# ---------------------------
# Grid Search for Hyperparameter Tuning
# ---------------------------
def grid_search_imputers(
        train_df,
        target_masked_df,
        numeric_cols,
        reference_full_df=None,
        random_state=42,
        verbose=0
):
    """

    """
    param_grids = {
        "BayesianRidge": {
        "estimator": [BayesianRidge()]
        },
        "KNN": {
            "estimator": [KNeighborsRegressor()],
            "estimator__n_neighbors": [3, 5, 7, 9],
            "estimator__weights": ["uniform", "distance"],
        },
        "RandomForest": {
            "estimator": [RandomForestRegressor(n_jobs=-1, random_state=random_state)],
            "estimator__n_estimators": [50, 100, 200],
            "estimator__max_depth": [None, 10, 20],
        },
    }

    all_results = []
    best_models = {}
    best_params = {}

    print("\n--- Starting Grid Search for Imputer Models ---\n")

    for model_name, grid in param_grids.items():
        print(f">>> Tuning {model_name} ...")
        best_rmse = np.inf
        best_param_set = None
        best_model_instance = None

        for params in ParameterGrid(grid):
            estimator = params.pop("estimator")
            estimator.set_params(**{k.replace("estimator__", ""): v for k, v in params.items()})

            _, _, metrics = iterative_imputer_once(
                estimator=estimator,
                train_df=train_df,
                target_masked_df=target_masked_df,
                numeric_cols=numeric_cols,
                reference_full_df=reference_full_df,
                random_state=random_state,
                verbose=verbose
            )

            all_results.append({
                "model": model_name,
                "params": params,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "total_time": metrics["total_time"]
            })

            if metrics["rmse"] < best_rmse:
                best_rmse = metrics["rmse"]
                best_param_set = params
                best_model_instance = estimator

        best_models[model_name] = best_model_instance
        best_params[model_name] = best_param_set

        print(f"Best {model_name}: RMSE={best_rmse:.4f} | Params={best_param_set}")

    grid_results_df = pd.DataFrame(all_results)

    print("\n--- Grid Search Summary ---\n")
    print(grid_results_df.groupby("model")[["rmse", "mae", "total_time"]].mean())

    return grid_results_df, best_models, best_params

# ---------------------------
# Retrain Models with B Parameters
# ---------------------------
def retrain_models_with_best_params(
        best_models,
        train_df,
        target_masked_df,
        numeric_cols,
        reference_full_df=None,
        random_state=42,
        verbose=0
):
    """

    """
    print("\n--- Retraining Best Models and Evaluating Performance ---\n")

    results_df, imputers, imputed_dfs = evaluate_imputer_candidates(
        candidates=best_models,
        train_df=train_df,
        target_masked_df=target_masked_df,
        numeric_cols=numeric_cols,
        reference_full_df=reference_full_df,
        random_state=random_state,
        verbose=verbose
    )

    print("\nFinal Evaluation of Tuned Models:")
    print(results_df[["model", "mae", "rmse", "total_time"]])

    return results_df, imputers, imputed_dfs



# ---------------------------
# Main Workflow
# ---------------------------
def main():
    correlation_threshold = 0.9  # The threshold is 90% for considering features as highly correlated
    missing_rate = 0.25  # 25% missing data to simulate

    filepath = r"C:\Users\marku\OneDrive - Aalborg Universitet\Githubs\P7_UTRY_Py\Mendeley Datasets\Body Measurements _ original_CSV.csv"
    df = load_and_clean_data(filepath)

    # Data overview
    data_overview(df)

    df = df.drop_duplicates()

    # Split dataset into train(70%)/val(15%)/test(15%) (by rows, i.e. people)
    train_df, val_df, test_df = split_dataset(df, train_frac=0.7, val_frac=0.15, test_frac=0.15, random_state=42)

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
    #train_df_trans, val_df_trans, test_df_trans = transform_and_scale(train_df, val_df, test_df, num_cols_no_gender)

    # Remove outliers first (then transform and scale)
    train_df_cleaned = mahalanobis_chi_outliers(train_df, length_cols, alpha=0.0001, remove=True)

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

    #train_df_cleaned = mahalanobis_chi_outliers(train_df_trans, length_cols, alpha=0.0001, remove=True)

    # Then fit transformer on cleaned data
    train_df_trans, val_df_trans, test_df_trans = transform_and_scale(train_df_cleaned, val_df, test_df,
                                                                      num_cols_no_gender)

    # Data overview after outlier removal
    #plot_plots(train_df_cleaned, num_cols_no_gender, length_cols)   # Some features show multimodal distributions after outlier removal, likely due population subgroups (e.g., Gender, Age)
    #shapiro_wilk_test(train_df_cleaned, num_cols_no_gender)
    #correlation_analysis(train_df_cleaned, num_cols_no_gender, correlation_threshold)

    train_full = train_df_cleaned

    val_full = val_df_trans
    val_masked_df, val_mask = simulate_missing_data(val_full, missing_rate=missing_rate, random_state=42, prefix="val")  # Simulate 20% missing data

    test_full = test_df_trans
    test_masked_df, test_mask = simulate_missing_data(test_full, missing_rate=missing_rate, random_state=42, prefix="test")  # Simulate 20% missing data

    # Candidate estimators to embed inside IterativeImputer
    candidates = {
        "BayesianRidge": BayesianRidge(),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "RandomForest": RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    }


    # We perform nested validation:
    # Evaluate candidates on validation masked set
    cv_results = cross_validate_imputers(
        candidates=candidates,
        full_train_df=train_full,
        numeric_cols=num_cols_no_gender,
        n_splits=5,
        missing_rate=missing_rate,
        random_state=42
    )

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
    )
    # Print the best model from CV (lowest RMSE)
    best_model_name = cv_results.sort_values("rmse").iloc[0]["model"]
    print(f"\nBest model from CV by RMSE: {best_model_name}")
    print("\nBest validation comparison between candidate(s) by RMSE:\n", results_df.sort_values('rmse').head())

    # Combined CV and validation results for easy comparison
    comparison_df = cv_results.merge(
        results_df[["model", "mae", "rmse", "total_time"]],
        on="model", suffixes=("_cv", "_val")
    ).sort_values("rmse_val")

    print("\n--- Combined CV vs Validation Summary ---\n")
    print(comparison_df)

    # Grid Search for Best Parameters ---
    grid_results_df, best_models, best_params = grid_search_imputers(
        train_df=train_full,
        target_masked_df=val_masked_df,
        numeric_cols=num_cols_no_gender,
        reference_full_df=val_full,
        random_state=42,
        verbose=0
    )

    print("\nBest parameters per model:\n", best_params)

    # Retrain and Evaluate the Best Models ---
    final_results_df, imputers, imputed_dfs = retrain_models_with_best_params(
        best_models=best_models,
        train_df=train_full,
        target_masked_df=val_masked_df,
        numeric_cols=num_cols_no_gender,
        reference_full_df=val_full,
        random_state=42,
        verbose=0
    )


if __name__ == "__main__":
    main()