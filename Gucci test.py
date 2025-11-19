import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer # Required for type hinting/checking
from Gucci import simulate_missing_data

# --- 1. Load trained artifacts ---
# Load trained imputer (IterativeImputer)
try:
    imputer = joblib.load("final_imputer.pkl")
    # Imputer contains the feature names it was trained on
    numeric_cols_no_gender = list(imputer.feature_names_in_)
except FileNotFoundError:
    print("Error: 'final_imputer.pkl' not found. Please run Gucci.py first.")
    exit()

# Load the fitted PowerTransformer (which also performed scaling)
try:
    # Ensure PowerTransformer is imported, although joblib handles the object
    transformer: PowerTransformer = joblib.load("transformer.pkl")
except FileNotFoundError:
    print("Error: 'transformer.pkl' not found. Please run Gucci.py first.")
    exit()

# --- 2. Hardcoded Input Data (Example) ---
data = {
    "Gender": [1],
    "Age": [30],
    "HeadCircumference": [np.nan], # Missing
    "ShoulderWidth": [18],
    "ChestWidth": [20],
    "Belly": [np.nan], # Missing
    "Waist": [14],
    "Hips": [22],
    "ArmLength": [22],
    "ShoulderToWaist": [np.nan], # Missing
    "WaistToKnee": [25],
    "LegLength": [22],
    "TotalHeight": [52],
}

# --- 3. Utility Function to Capture Mask (Modified from Gucci.py for deployment use) ---
def get_missing_mask(df, numeric_cols):
    """
    Creates a boolean mask indicating which values are NaN.
    """
    mask = df[numeric_cols].isna()
    return mask

input_df = pd.DataFrame(data)

# The deployment input is already an "input_df" with NaNs; we don't need the simulate_missing_data helper
# We need to correctly encode the 'Gender' column if it's present and not already 0/1,
# but based on Gucci.py, the final imputer was trained on Gender=0/1 and did not use it for imputation.

# The imputer was trained only on numeric_cols_no_gender.

# --- 4. Pre-processing for Imputation ---

# 4a. Capture the mask of the ORIGINAL (untransformed) data.
# This mask tells us which values were imputed and need reversal later.
mask_original = get_missing_mask(input_df, numeric_cols_no_gender)

# 4b. Temporarily align and scale the input data (to the state expected by the imputer)
# The PowerTransformer handles NaNs internally by ignoring them during transformation,
# but IterativeImputer *requires* the NaNs to be present for the 'transform' step.
# For simplicity, we directly pass the original data (with NaNs) to imputer.transform,
# which uses an internal SimpleImputer (mean/median/most_frequent) before starting iteration.

input_df_imputer_ready = input_df[numeric_cols_no_gender]

# --- 5. Imputation and Inverse Transformation ---

# 5a. Impute missing values (results are scaled/transformed)
# The output 'imputed_scaled_array' has no NaNs, and all values are in the scaled/transformed space.
imputed_scaled_array = imputer.transform(input_df_imputer_ready)
imputed_scaled_df = pd.DataFrame(
    imputed_scaled_array,
    columns=numeric_cols_no_gender,
    index=input_df.index
)

# 5b. Inverse Transform the ENTIRE imputed dataset back to the original scale
# This is necessary because both original non-missing values and imputed values are currently scaled.
imputed_original_scale_array = transformer.inverse_transform(imputed_scaled_array)
imputed_original_scale_df = pd.DataFrame(
    imputed_original_scale_array,
    columns=numeric_cols_no_gender,
    index=input_df.index
)

# --- 6. Selective Update (The Core Fix) ---

# We only want to replace the values that were originally missing (where the mask is True).
# The original data (with NaNs) for these columns.
final_result_df = input_df.copy()

# Iterate over the columns where imputation occurred
for col in numeric_cols_no_gender:
    # Identify the indices where the value was missing (and thus imputed)
    indices_to_update = mask_original[col]

    # For these indices, replace the NaN in the final result with the
    # value from the imputed_original_scale_df.
    final_result_df.loc[indices_to_update, col] = imputed_original_scale_df.loc[indices_to_update, col]

# --- 7. Output Result ---
print("--- Input Data with Missing Values ---")
print(input_df.to_string())

print("\n--- Imputed Data (Only imputed entries are inverse-transformed) ---")
print(final_result_df.to_string())

# Verify that non-missing entries remain unchanged and imputed entries are now filled
print("\nVerification (Only imputed values have changed, and they are now filled):")
for col in numeric_cols_no_gender:
    if input_df[col].isnull().any():
        imputed_count = final_result_df[col].isnull().sum()
        print(f"Column '{col}' - NaN Count: {input_df[col].isnull().sum()} (before) -> {imputed_count} (after)")
        if imputed_count == 0:
            print(f"  > Imputation successful for {col}")