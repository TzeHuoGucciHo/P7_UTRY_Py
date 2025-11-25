import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, ParameterGrid
from sklearn.preprocessing import PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, probplot

# ---------------------------
# Data Loading and Splitting
# ---------------------------
def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    return df

def convert_inches_to_cm(df, numeric_cols):
    cols_to_convert = [col for col in numeric_cols if col not in ['Age', 'Gender']]

    print(f"\nConverting {len(cols_to_convert)} columns from inches to cm...")

    df[cols_to_convert] = df[cols_to_convert] * 2.54

    return df

def split_dataset(df, train_frac=0.7, val_frac=0.15, test_frac=0.15, random_state=42):
    train_df, temp_df = train_test_split(
        df,
        train_size=train_frac,
        shuffle=True,
        random_state=random_state
    )
    val_size = val_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_size,
        shuffle=True,
        random_state=random_state
    )
    return train_df, val_df, test_df

# ---------------------------
# Exploratory Data Analysis (EDA)
# ---------------------------
def data_overview(df, numeric_cols, cat_cols, title=""):
    print(f"\n===== {title.upper()} =====\n")

    print("Shape:", df.shape)
    print("\nColumn types:\n", df.dtypes)

    print("\nMissing values per column:\n", df.isnull().sum())
    print("\nNumber of duplicates:", df.duplicated().sum())

    print("\nNumeric feature statistics:\n", df[numeric_cols].describe())
    for col in numeric_cols:
        print(f"{col} skewness: {df[col].skew():.2f}")

    print("\nCategorical feature counts:")
    for col in cat_cols:
        print(f"\n{col}:\n", df[col].value_counts())

    num_features = len(numeric_cols)

    # ---------------------------
    # FIGURE 1: Histograms
    # ---------------------------
    plt.figure(figsize=(12, 4 * num_features))

    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(num_features, 1, i)
        sns.histplot(
            df[col],
            bins=30,
            kde=True,
            edgecolor='black',
            linewidth=0.5
        )
        plt.axvline(df[col].mean(), color='red', linestyle='--', linewidth=1, label='Mean')
        plt.axvline(df[col].median(), color='green', linestyle='-', linewidth=1, label='Median')

        plt.title(f"{col} Distribution", fontsize=14)
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle(f"{title} — Histograms", fontsize=18)
    plt.show()

    # ---------------------------
    # FIGURE 2: Boxplots
    # ---------------------------
    plt.figure(figsize=(12, 2.5 * num_features))

    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(num_features, 1, i)
        sns.boxplot(x=df[col], color='lightgreen')
        plt.title(f"{col} Boxplot")
        plt.xlabel(col)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle(f"{title} — Boxplots", fontsize=18)
    plt.show()

    # ---------------------------
    # FIGURE 3: Q-Q Plots
    # ---------------------------
    plt.figure(figsize=(12, 4 * num_features))

    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(num_features, 1, i)
        probplot(df[col], dist="norm", plot=plt)
        plt.title(f"{col} Q-Q Plot")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle(f"{title} — Q-Q Plots (Normality Check)", fontsize=18)
    plt.show()

    # ---------------------------
    # FIGURE 4: Correlation Heatmap
    # ---------------------------
    plt.figure(figsize=(12, 10))

    corr = df[numeric_cols].corr()
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.7}
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle(f"{title} — Correlation Heatmap", fontsize=18)
    plt.show()

    # ---------------------------
    # Shapiro-Wilk summary
    # ---------------------------
    print("\nShapiro-Wilk Normality Test (p > 0.05 means roughly normal):")
    for col in numeric_cols:
        stat, p = shapiro(df[col])
        print(f"{col}: p={p:.3f} -> {'Normal-ish' if p > 0.05 else 'Not normal'}")

# ---------------------------
# Data Preprocessing
# ---------------------------
def remove_outliers_binned_iqr(df, group_col, value_cols, bins=10, threshold=3.0):

    df_clean = df.copy()
    initial_rows = len(df_clean)
    indices_to_drop = set()

    df_clean['temp_group_bin'] = pd.cut(df_clean[group_col], bins=bins)

    print(f"\n--- Outlier Detection (Grouped by Binned '{group_col}', Threshold={threshold}) ---")
    print(f"Binning strategy: {bins}")

    for col in value_cols:
        grouper = df_clean.groupby('temp_group_bin', observed=True)[col]

        q1 = grouper.transform(lambda x: x.quantile(0.25))
        q3 = grouper.transform(lambda x: x.quantile(0.75))
        iqr = q3 - q1

        lower = q1 - (threshold * iqr)
        upper = q3 + (threshold * iqr)

        outlier_mask = (df_clean[col] < lower) | (df_clean[col] > upper)
        outlier_indices = df_clean[outlier_mask].index

        count = len(outlier_indices)
        indices_to_drop.update(outlier_indices)

        if count > 0:
            print(f"   {col}: detected {count} outliers")

    df_clean = df_clean.drop(columns=['temp_group_bin'])
    df_clean = df_clean.drop(index=list(indices_to_drop))

    total_dropped = initial_rows - len(df_clean)
    percent_dropped = (total_dropped / initial_rows) * 100

    print("-" * 40)
    print(f"Total rows removed: {total_dropped} ({percent_dropped:.2f}%)")
    print(f"Final shape: {df_clean.shape}")
    print("-" * 40)

    return df_clean

def transform_and_scale(train_df, val_df, test_df, numeric_cols, standardize=True):
    transformer = PowerTransformer(method='yeo-johnson', standardize=standardize)
    train_transformed = transformer.fit_transform(train_df[numeric_cols])
    val_transformed = transformer.transform(val_df[numeric_cols])
    test_transformed = transformer.transform(test_df[numeric_cols])

    train_df_trans, val_df_trans, test_df_trans = train_df.copy(), val_df.copy(), test_df.copy()
    train_df_trans[numeric_cols] = train_transformed
    val_df_trans[numeric_cols] = val_transformed
    test_df_trans[numeric_cols] = test_transformed

    return train_df_trans, val_df_trans, test_df_trans, transformer

def simulate_missing_data(df, missing_rate=0.2, random_state=42):
    np.random.seed(random_state)
    df_masked = df.copy()
    mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in ['Gender']]
    total_cells = len(df) * len(numeric_cols)
    total_to_mask = int(total_cells * missing_rate)

    all_positions = [(i, col) for i in df.index for col in numeric_cols]
    selected_positions = np.random.choice(len(all_positions), size=total_to_mask, replace=False)
    for idx in selected_positions:
        row, col = all_positions[idx]
        df_masked.at[row, col] = np.nan
        mask.at[row, col] = True

    return df_masked, mask

# ---------------------------
# Iterative Imputer
# ---------------------------
def iterative_imputer(train_df, target_df, model, numeric_cols, max_iterations, verbose=False):
    if verbose:
        print(f"\nImputing with model: {model.__class__.__name__}")
    start_time = time.time()

    imputer = IterativeImputer(
        estimator=model,
        max_iter=max_iterations,
        random_state=42,
        initial_strategy='mean'
    )
    imputer.fit(train_df[numeric_cols])
    imputed_array = imputer.transform(target_df[numeric_cols])
    runtime = time.time() - start_time

    imputed_df = target_df.copy()
    imputed_df[numeric_cols] = imputed_array

    if verbose:
        print(f"Imputation complete in {runtime:.2f} seconds.")
    return {"model_name": model.__class__.__name__, "imputed_df": imputed_df, "runtime": runtime}

# ---------------------------
# Evaluate Imputation
# ---------------------------
def evaluate_imputation(imputed_df, target_df_full, target_df_mask, numeric_cols):
    mae_list, rmse_list = [], []
    for col in numeric_cols:
        masked_indices = target_df_mask[col]
        true_vals = target_df_full.loc[masked_indices, col]
        imputed_vals = imputed_df.loc[masked_indices, col]
        mae_list.append(mean_absolute_error(true_vals, imputed_vals))
        rmse_list.append(np.sqrt(mean_squared_error(true_vals, imputed_vals)))
    return {"MAE": np.mean(mae_list), "RMSE": np.mean(rmse_list)}

# ---------------------------
# Model Comparison
# ---------------------------
def compare_models(models, train_df, target_df, target_df_full, target_df_mask, numeric_cols):
    results = []
    for model in models:
        impute_result = iterative_imputer(train_df, target_df, model, numeric_cols, max_iterations=50)
        metrics = evaluate_imputation(impute_result["imputed_df"], target_df_full, target_df_mask, numeric_cols)
        results.append({"Model": impute_result["model_name"], "MAE": metrics["MAE"], "RMSE": metrics["RMSE"], "Runtime_sec": impute_result["runtime"]})
    results_df = pd.DataFrame(results)
    results_df['score'] = 1 / results_df['MAE'] / results_df['Runtime_sec']
    return results_df.sort_values(by='score', ascending=False).reset_index(drop=True)

# ---------------------------
# Cross-validation
# ---------------------------
def cross_validation(train_full, numeric_cols, models, k=5, missing_rate=0.25, max_iterations=50, random_state=42, fixed_missingness=True):
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    results = {model.__class__.__name__: {'MAE': [], 'RMSE': []} for model in models}
    full_masked_df, full_mask_values = simulate_missing_data(train_full, missing_rate=missing_rate, random_state=random_state) if fixed_missingness else (None, None)

    for fold_idx, (train_index, val_index) in enumerate(kf.split(train_full), start=1):
        train_fold = train_full.iloc[train_index].copy()
        val_fold_full = train_full.iloc[val_index].copy()
        val_fold_masked, val_mask_values = (full_masked_df.iloc[val_index].copy(), full_mask_values.iloc[val_index].copy()) if fixed_missingness else simulate_missing_data(val_fold_full, missing_rate=missing_rate, random_state=random_state + fold_idx)

        for model in models:
            impute_result = iterative_imputer(train_fold, val_fold_masked, model, numeric_cols, max_iterations)
            metrics = evaluate_imputation(impute_result['imputed_df'], val_fold_full, val_mask_values, numeric_cols)
            results[model.__class__.__name__]['MAE'].append(metrics['MAE'])
            results[model.__class__.__name__]['RMSE'].append(metrics['RMSE'])

    summary = []
    for model_name, metrics_dict in results.items():
        summary.append({
            'Model': model_name,
            'MAE_mean': np.mean(metrics_dict['MAE']),
            'MAE_std': np.std(metrics_dict['MAE']),
            'RMSE_mean': np.mean(metrics_dict['RMSE']),
            'RMSE_std': np.std(metrics_dict['RMSE'])
        })
    return pd.DataFrame(summary).sort_values('MAE_mean').reset_index(drop=True)

# ---------------------------
# Grid Search
# ---------------------------
def grid_search(train_full, numeric_cols, model_param_grids, k=5, missing_rate=0.25, max_iterations=50, random_state=42):
    all_results = []
    for model_name, param_grid in model_param_grids.items():
        for params in ParameterGrid(param_grid):
            if model_name == "BayesianRidge": model = BayesianRidge(**params)
            elif model_name == "KNeighborsRegressor": model = KNeighborsRegressor(**params)
            elif model_name == "RandomForestRegressor": model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
            else: raise ValueError(f"Unknown model name: {model_name}")
            cv_result = cross_validation(train_full, numeric_cols, models=[model], k=k, missing_rate=missing_rate, max_iterations=max_iterations, random_state=random_state)
            cv_result["params"] = [params] * len(cv_result)
            all_results.append(cv_result)
    return pd.concat(all_results, ignore_index=True).sort_values(by="MAE_mean").reset_index(drop=True)

# ---------------------------
# Main Workflow
# ---------------------------
def main():
    filepath = r"C:\Uni\MED7\Semester project\P7_UTRY_Py\Mendeley Datasets\Body Measurements _ original_CSV.csv"
    df = load_and_clean_data(filepath)
    df = df.drop_duplicates()

    all_num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = convert_inches_to_cm(df, all_num_cols)

    train_df, val_df, test_df = split_dataset(df)

    num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols_no_gender = [col for col in num_cols if col not in ['Gender']]
    cat_cols = ['Gender']
    cat_mode_imputer = SimpleImputer(strategy='most_frequent')
    cat_mode_imputer.fit(train_df[cat_cols])

    # data_overview(df, num_cols, cat_cols, title="Full Dataset")

    print("\nTrain Missing values:\n", train_df.isnull().sum())
    print("\nVal Missing values:\n", val_df.isnull().sum())
    print("\nTest Missing values:\n", test_df.isnull().sum())

    # ---------------------------
    # Data Cleaning and Preprocessing
    # ---------------------------
    num_mean_imputer = SimpleImputer(strategy='mean')
    num_mean_imputer.fit(train_df[num_cols_no_gender])

    train_df[num_cols_no_gender] = num_mean_imputer.transform(train_df[num_cols_no_gender])
    val_df[num_cols_no_gender] = num_mean_imputer.transform(val_df[num_cols_no_gender])
    test_df[num_cols_no_gender] = num_mean_imputer.transform(test_df[num_cols_no_gender])

    train_df[cat_cols] = cat_mode_imputer.transform(train_df[cat_cols])
    val_df[cat_cols] = cat_mode_imputer.transform(val_df[cat_cols])
    test_df[cat_cols] = cat_mode_imputer.transform(test_df[cat_cols])

    print("\nTrain Missing values:\n", train_df.isnull().sum())
    print("\nVal Missing values:\n", val_df.isnull().sum())
    print("\nTest Missing values:\n", test_df.isnull().sum())

    mapping = {1: 0, 2: 1}
    for df_ in [train_df, val_df, test_df]:
        df_['Gender'] = df_['Gender'].map(mapping)

    train_df = remove_outliers_binned_iqr(
        train_df,
        group_col='Age',
        value_cols=num_cols_no_gender,
        bins=10,
        threshold=6.0
    )
    #data_overview(train_df, num_cols, cat_cols, title="Post-outlier-removal train_df")

    train_df_trans, val_df_trans, test_df_trans, transformer = transform_and_scale(train_df, val_df, test_df, num_cols_no_gender, standardize=True)
    #data_overview(train_df_trans, num_cols, cat_cols, title="Post-transformation train_df_trans")

    val_masked_df, val_mask_values = simulate_missing_data(val_df_trans, missing_rate=0.25)
    test_masked_df, test_mask_values = simulate_missing_data(test_df_trans, missing_rate=0.25)

    # ---------------------------
    # Model Comparison and Selection
    # ---------------------------
    candidates = [
        BayesianRidge(),
        KNeighborsRegressor(n_neighbors=5),
        RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    ]

    compare_models(candidates, train_df_trans, val_masked_df, val_df_trans, val_mask_values, num_cols_no_gender)
    cv_results = cross_validation(train_df_trans, num_cols_no_gender, candidates)

    param_grids = {
        "BayesianRidge": {"alpha_1": [1e-6, 1e-5, 1e-4], "alpha_2": [1e-6, 1e-5, 1e-4], "lambda_1": [1e-6, 1e-5, 1e-4], "lambda_2": [1e-6, 1e-5, 1e-4]},
        "KNeighborsRegressor": {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]},
        "RandomForestRegressor": {"n_estimators": [50, 100], "max_depth": [None, 10, 20], "min_samples_split": [2, 5]}
    }

    grid_results = grid_search(train_df_trans, num_cols_no_gender, param_grids, max_iterations=30)
    best_config = grid_results.iloc[0]

    if best_config["Model"] == "BayesianRidge": best_model = BayesianRidge(**best_config["params"])
    elif best_config["Model"] == "KNeighborsRegressor": best_model = KNeighborsRegressor(**best_config["params"])
    elif best_config["Model"] == "RandomForestRegressor": best_model = RandomForestRegressor(**best_config["params"], random_state=42, n_jobs=-1)

    train_val_full = pd.concat([train_df_trans, val_df_trans], ignore_index=True)
    final_imputer = IterativeImputer(estimator=best_model, max_iter=50, random_state=42, initial_strategy='mean')
    final_imputer.fit(train_val_full[num_cols_no_gender])

    # ---------------------------
    # Model Deployment
    # ---------------------------
    joblib.dump(transformer, "transformer.pkl")
    joblib.dump(final_imputer, "final_imputer.pkl")

if __name__ == "__main__":
    main()
