import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer

def convert_inches_to_cm(df, numeric_cols):
    cols_to_convert = [col for col in numeric_cols if col not in ['Age', 'Gender']]

    print(f"\nConverting {len(cols_to_convert)} columns from inches to cm...")

    df[cols_to_convert] = df[cols_to_convert] * 2.54

    return df

# Load saved models and feature names
imputer = joblib.load("final_imputer.pkl")
# Use the feature names from the imputer to define the order of columns
numeric_cols_no_gender = list(imputer.feature_names_in_)
transformer: PowerTransformer = joblib.load("transformer.pkl")

# --- NEW INPUT DATA (Assumed to be in Inches based on the values) ---
data = {
    "Gender": [1],
    "Age": [23],
    "HeadCircumference": [22.9],
    "ShoulderWidth": [17.7],
    "ChestWidth": [np.nan],
    "Belly": [np.nan],
    "Waist": [42.3],
    "Hips": [47.6],
    "ArmLength": [np.nan],
    "ShoulderToWaist": [np.nan],
    "WaistToKnee": [np.nan],
    "LegLength": [np.nan],
    "TotalHeight": [75.2],
}

def get_missing_mask(df, cols):
    return df[cols].isna()

input_df = pd.DataFrame(data)

# --- CRITICAL NEW STEP: CONVERT INPUT DATA TO CM ---
# We need ALL numeric columns from the input data to pass to the conversion function
all_numeric_cols = [col for col in input_df.columns if input_df[col].dtype != 'object']
input_df = convert_inches_to_cm(input_df, all_numeric_cols)
# ----------------------------------------------------

mask_original = get_missing_mask(input_df, numeric_cols_no_gender)
input_scaled = transformer.transform(input_df[numeric_cols_no_gender])

imputed_scaled_array = imputer.transform(input_scaled)
imputed_original_scale_array = transformer.inverse_transform(imputed_scaled_array)

imputed_original_scale_df = pd.DataFrame(
    imputed_original_scale_array,
    columns=numeric_cols_no_gender,
    index=input_df.index
)

final_result_df = input_df.copy()
for col in numeric_cols_no_gender:
    # Fill the NaNs from the original *converted* input with the imputed values
    final_result_df.loc[mask_original[col], col] = imputed_original_scale_df[col]

print("--- Input Data with Missing Values (Converted to CM) ---")
print(input_df.to_string())

print("\n--- Imputed Data (Only imputed entries are in CM scale) ---")
print(final_result_df.to_string())

print("\nVerification:")
for col in numeric_cols_no_gender:
    if mask_original[col].any():
        print(f"{col}: {mask_original[col].sum()} missing values found → {final_result_df[col].isna().sum()} missing values remain (0 means success)")


