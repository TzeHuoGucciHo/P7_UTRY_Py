import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer

# Load saved models and feature names
imputer = joblib.load("final_imputer.pkl")
# Use the feature names from the imputer to define the order of columns
numeric_cols_no_gender = list(imputer.feature_names_in_)
transformer: PowerTransformer = joblib.load("transformer.pkl")

# --- NEW INPUT DATA (In CM) ---
data = {
    "Gender": [1.0],
    "Age": [23.0],
    "HeadCircumference": [69],
    "ShoulderWidth": [54],
    "ChestWidth": [41],
    "Belly": [117],
    "Waist": [np.nan],
    "Hips": [np.nan],
    "ArmLength": [63],
    "ShoulderToWaist": [32],
    "WaistToKnee": [68],
    "LegLength": [np.nan],
    "TotalHeight": [190.0]
}

# In CM
true_val = {
    "Gender": 1.0,
    "Age": 23.0,
    "HeadCircumference": 69,
    "ShoulderWidth": 54,
    "ChestWidth": 41,
    "Belly": 117,
    "Waist": 104,
    "Hips": 122,
    "ArmLength": 63,
    "ShoulderToWaist": 32,
    "WaistToKnee": 68,
    "LegLength": 88,
    "TotalHeight": 190.0
}

def get_missing_mask(df, cols):
    return df[cols].isna()

input_df = pd.DataFrame(data)

true_val = pd.DataFrame([true_val])

all_numeric_cols = [col for col in input_df.columns if input_df[col].dtype != 'object']
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
    final_result_df.loc[mask_original[col], col] = imputed_original_scale_df[col]

print("--- Input Data with Missing Values (Converted to CM) ---")
print(input_df.to_string())

print("\n--- Imputed Data (Only imputed entries are in CM scale) ---")
print(final_result_df.to_string())

print("\n--- True Val ---")
print(true_val.to_string())

