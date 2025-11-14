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
import joblib


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
def simulate_missing_data(df, missing_rate=0.2, random_state=42, prefix="", verbose=False):
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

    # Randomly select (row, col) pairs without replacement
    all_positions = [(i, col) for i in df.index for col in numeric_cols]
    selected_positions = np.random.choice(len(all_positions), size=total_to_mask, replace=False)

    for idx in selected_positions:
        row, col = all_positions[idx]
        df_masked.at[row, col] = np.nan
        mask.at[row, col] = True

    if verbose:
        print(f"{prefix}: Missingness simulated ({missing_rate * 100:.0f}% of numeric cells)")

    return df_masked, mask

# ---------------------------
# Iterative Imputer
# ---------------------------
def iterative_imputer(train_df, target_df, model, numeric_cols, max_iterations, verbose=False):
    """

    """
    if verbose:
        print(f"\nImputing with model: {model.__class__.__name__}")

    start_time = time.time()

    # Initialize IterativeImputer with given model
    imputer = IterativeImputer(
        estimator=model,
        max_iter=max_iterations,
        random_state=42,
        initial_strategy='mean'
    )

    # Fit imputer on training data
    imputer.fit(train_df[numeric_cols])

    # Impute missing values in target dataset
    imputed_array = imputer.transform(target_df[numeric_cols])

    runtime = time.time() - start_time

    # Create a copy of the target dataframe with the imputed values.
    imputed_df = target_df.copy()
    imputed_df[numeric_cols] = imputed_array

    if verbose:
        print(f"Imputation complete in {runtime:.2f} seconds.")

    return {
        "model_name": model.__class__.__name__,
        "imputed_df": imputed_df,
        "runtime": runtime
    }

# ---------------------------
# Evaluate Imputer
# ---------------------------
def evaluate_imputation(imputed_df, target_df_full, target_df_mask, numeric_cols):

    mae_list, rmse_list = [], []

    for col in numeric_cols:
        masked_indices = target_df_mask[col]  # This should already be aligned to the dataframe

        true_vals = target_df_full.loc[masked_indices, col]
        imputed_vals = imputed_df.loc[masked_indices, col]

        mae = mean_absolute_error(true_vals, imputed_vals)
        rmse = math.sqrt(mean_squared_error(true_vals, imputed_vals))
        mae_list.append(mae)
        rmse_list.append(rmse)

    # Aggregate metrics across all numeric columns
    overall_mae = np.mean(mae_list)
    overall_rmse = np.mean(rmse_list)

    results = {
        "MAE": overall_mae,
        "RMSE": overall_rmse
    }

    return results

def compare_models(models, train_df, target_df, target_df_full, target_df_mask, numeric_cols, verbose=True):
    """

    """
    results = []

    for model in models:
        impute_result = iterative_imputer(train_df, target_df, model, numeric_cols, max_iterations=50, verbose=verbose)
        imputed_df = impute_result['imputed_df']
        runtime = impute_result['runtime']

        metrics = evaluate_imputation(imputed_df, target_df_full, target_df_mask, numeric_cols)

        results.append({
            "Model": impute_result['model_name'],
            "MAE": metrics['MAE'],
            "RMSE": metrics['RMSE'],
            "Runtime_sec": runtime
        })

    # Convert results list to DataFrame for easy comparison
    results_df = pd.DataFrame(results)
    results_df['score'] = 1 / results_df['MAE'] / results_df['Runtime_sec']
    results_df = results_df.sort_values(by='score', ascending=False)
    results_df.reset_index(drop=True, inplace=True)

    if verbose:
        print("\nSummary of Model Comparison:")
        print(results_df)

    return results_df

def cross_validation(train_full, numeric_cols, models, k=5, missing_rate=0.25, max_iterations=50, random_state=42, fixed_missingness=True, verbose=False):
    """
    Performs k-fold cross-validation for iterative imputation models,
    optionally using a fixed missingness mask across folds.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    results = {model.__class__.__name__: {'MAE': [], 'RMSE': []} for model in models}

    # Pre-generate fixed missingness for the entire dataset if requested
    if fixed_missingness:
        full_masked_df, full_mask_values = simulate_missing_data(train_full, missing_rate=missing_rate, random_state=random_state, verbose=verbose)

    fold_idx = 1
    for train_index, val_index in kf.split(train_full):
        if verbose:
            print(f"\n===== Fold {fold_idx} =====")

        train_fold = train_full.iloc[train_index].copy()
        val_fold_full = train_full.iloc[val_index].copy()

        # Apply missingness
        if fixed_missingness:
            val_fold_masked = full_masked_df.iloc[val_index].copy()
            val_mask_values = full_mask_values.iloc[val_index].copy()
        else:
            val_fold_masked, val_mask_values = simulate_missing_data(
                val_fold_full,
                missing_rate=missing_rate,
                random_state=random_state + fold_idx,
                verbose=verbose
            )

        # Impute & evaluate
        for model in models:
            impute_result = iterative_imputer(train_fold, val_fold_masked, model, numeric_cols, max_iterations, verbose=verbose)
            imputed_df = impute_result['imputed_df']
            metrics = evaluate_imputation(imputed_df, val_fold_full, val_mask_values, numeric_cols)

            results[model.__class__.__name__]['MAE'].append(metrics['MAE'])
            results[model.__class__.__name__]['RMSE'].append(metrics['RMSE'])

        fold_idx += 1

    # Aggregate results
    summary = []
    for model_name, metrics_dict in results.items():
        summary.append({
            'Model': model_name,
            'MAE_mean': np.mean(metrics_dict['MAE']),
            'MAE_std': np.std(metrics_dict['MAE']),
            'RMSE_mean': np.mean(metrics_dict['RMSE']),
            'RMSE_std': np.std(metrics_dict['RMSE'])
        })

    results_df = pd.DataFrame(summary).sort_values('MAE_mean').reset_index(drop=True)

    if verbose:
        print("\n===== CV Summary =====")
        print(results_df)

    return results_df

# ---------------------------
# Grid Search for Hyperparameters
# ---------------------------
def grid_search(
    train_full,
    numeric_cols,
    model_param_grids,
    k=5,
    missing_rate=0.25,
    max_iterations=50,
    random_state=42,
    verbose=False,
): # Verbose=True will detail logs of each configuration
    """
    Performs grid search with cross-validation for each model and hyperparameter combination.
    Records both performance metrics and runtime for each configuration.
    """
    all_results = []

    print("\n=== Starting Grid Search ===")
    total_start = time.time()

    for model_name, param_grid in model_param_grids.items():
        print(f"\n▶ Model: {model_name} ({len(ParameterGrid(param_grid))} combinations)")

        for i, params in enumerate(ParameterGrid(param_grid), start=1):
            if verbose:
                print(f"  - [{i}] Params: {params}")
            else:
                print(f"  - [{i}/{len(ParameterGrid(param_grid))}] Running...", end="\r")

            # Instantiate model
            if model_name == "BayesianRidge":
                model = BayesianRidge(**params)
            elif model_name == "KNeighborsRegressor":
                model = KNeighborsRegressor(**params)
            elif model_name == "RandomForestRegressor":
                model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
            else:
                raise ValueError(f"Unknown model name: {model_name}")

            # Measure runtime per config
            start = time.time()

            cv_result = cross_validation(
                train_full=train_full,
                numeric_cols=numeric_cols,
                models=[model],
                k=k,
                missing_rate=missing_rate,
                max_iterations=max_iterations,
                random_state=random_state,
                fixed_missingness=True,
                verbose=verbose
            )

            end = time.time()
            runtime = end - start
            cv_result["params"] = [params] * len(cv_result)
            cv_result["runtime_sec"] = runtime
            all_results.append(cv_result)

    total_time = time.time() - total_start

    final_results = pd.concat(all_results, ignore_index=True)
    final_results = final_results.sort_values(by="MAE_mean").reset_index(drop=True)

    print("\n=== Grid Search Summary ===")

    # Aggregate per model
    summary = (
        final_results.groupby("Model")
        .apply(lambda g: g.loc[g["MAE_mean"].idxmin(), ["MAE_mean", "RMSE_mean", "runtime_sec", "params"]])
        .reset_index()
    )

    print("\nBest configuration per model:")
    print(summary.to_string(index=False))

    # Top N configs overall
    top_n = 5
    print(f"\nTop {top_n} overall configurations:")
    print(
        final_results.nsmallest(top_n, "MAE_mean")[["Model", "MAE_mean", "RMSE_mean", "runtime_sec", "params"]]
        .to_string(index=False)
    )

    print(f"\nTotal grid search time: {total_time/60:.2f} minutes")

    return final_results



# ---------------------------
# Main Workflow
# ---------------------------
def main():
    correlation_threshold = 0.9  # The threshold is 90% for considering features as highly correlated
    missing_rate = 0.25  # 25% missing data to simulate

    filepath = r"Mendeley Datasets/Body Measurements _ original_CSV.csv"
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

    train_full = train_df_trans.copy()

    val_full = val_df_trans.copy()
    test_full = test_df_trans.copy()

    # Simulate missingness on validation and test (we'll evaluate only on masked entries)
    val_masked_df, val_mask_values = simulate_missing_data(val_full, missing_rate=missing_rate, random_state=42, prefix="val", verbose=False)
    test_masked_df, test_mask_values = simulate_missing_data(test_full, missing_rate=missing_rate, random_state=42, prefix="test", verbose=False)
    # _masked_df is the dataset with missing values.
    # _mask_values are which values are artificially missing.

    candidates = [
        BayesianRidge(),
        KNeighborsRegressor(n_neighbors=5),
        RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    ]

#    compare_models_results_df = compare_models(
#        models=candidates,
#        train_df=train_full,
#        target_df=val_masked_df,
#        target_df_full=val_full,
#        target_df_mask=val_mask_values,
#        numeric_cols=num_cols_no_gender,
#        verbose=True
#    )

    cv_results = cross_validation(
        train_full=train_full,
        numeric_cols=num_cols_no_gender,
        models=candidates,
        k=5,
        missing_rate=missing_rate,
        max_iterations=50,
        random_state=42,
        fixed_missingness=True,
        verbose=False
    )

    param_grids = {
        "BayesianRidge": {
            "alpha_1": [1e-6, 1e-5, 1e-4],
            "alpha_2": [1e-6, 1e-5, 1e-4],
            "lambda_1": [1e-6, 1e-5, 1e-4],
            "lambda_2": [1e-6, 1e-5, 1e-4],
        },
        "KNeighborsRegressor": {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform", "distance"],
        },
        "RandomForestRegressor": {
            "n_estimators": [50, 100],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
        }
    }

    grid_results = grid_search(
        train_full=train_full,
        numeric_cols=num_cols_no_gender,
        model_param_grids=param_grids,
        k=5,
        missing_rate=missing_rate,
        max_iterations=30,
        verbose=False
    )

    best_config = (
        grid_results
        .sort_values("MAE_mean")
        .iloc[0]
    )
    print("Best model and parameters:")
    print(best_config)

    best_model_name = best_config["Model"]
    best_params = best_config["params"]

    if best_model_name == "BayesianRidge":
        best_model = BayesianRidge(**best_params)
    elif best_model_name == "KNeighborsRegressor":
        best_model = KNeighborsRegressor(**best_params)
    elif best_model_name == "RandomForestRegressor":
        best_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)

    val_impute_result = iterative_imputer(
        train_df=train_full,
        target_df=val_masked_df,
        model=best_model,
        numeric_cols=num_cols_no_gender,
        max_iterations=50,
        verbose=True
    )

    val_metrics = evaluate_imputation(
        imputed_df=val_impute_result["imputed_df"],
        target_df_full=val_full,
        target_df_mask=val_mask_values,
        numeric_cols=num_cols_no_gender
    )

    print("\nValidation Performance:")
    print(val_metrics)

    train_val_full = pd.concat([train_full, val_full], ignore_index=True)

    test_impute_result = iterative_imputer(
        train_df=train_val_full,
        target_df=test_masked_df,
        model=best_model,
        numeric_cols=num_cols_no_gender,
        max_iterations=50,
        verbose=True
    )

    test_metrics = evaluate_imputation(
        imputed_df=test_impute_result["imputed_df"],
        target_df_full=test_full,
        target_df_mask=test_mask_values,
        numeric_cols=num_cols_no_gender
    )

    print("\nTest Performance:")
    print(test_metrics)

    # Save the full iterative imputer (which contains the trained estimator)
    # joblib.dump(best_model, "best_model.pkl")

    # Save the final imputer fitted on train+val
    final_imputer = IterativeImputer(
        estimator=best_model,
        max_iter=50,
        random_state=42,
        initial_strategy='mean'
    )
    final_imputer.fit(train_val_full[num_cols_no_gender])


    joblib.dump(final_imputer, "final_imputer.pkl")

if __name__ == "__main__":
    main()