import joblib
import pandas as pd
import numpy as np
import json
import sys
import time
from typing import Dict, Any


# ---------------------------
# Helper to convert NumPy types to Python types for clean JSON
# ---------------------------
def default_converter(obj: Any) -> Any:
    """Converts NumPy types to standard Python types for JSON serialization."""
    if isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    if isinstance(obj, np.int64) or isinstance(obj, np.int32):
        return int(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')


# Initialize global timers and result storage
t_all = time.perf_counter()
timings: Dict[str, float] = {}
final_measurements: Dict[str, Any] = {}

# --- Configuration (MUST MATCH Imputer Training) ---
# NOTE: These columns must exactly match the features used when you trained final_imputer.pkl.
IMPUTER_FEATURES = [
    'ShoulderWidth', 'ChestWidth', 'Waist', 'Hips', 'TotalHeight',
    # Add any other required features here, e.g., 'Age', 'Weight', etc.
]
# --- End Configuration ---


# ---------------------------
# 1. SETUP AND INPUT LOADING (Ignores bad ML data)
# ---------------------------
t0 = time.perf_counter()

try:
    if len(sys.argv) < 2:
        raise IndexError("Missing input JSON file path argument (from Script 1 output).")

    input_json_path = sys.argv[1]

    # Load the raw measurements from cli.py (Script 1)
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data_from_cli = json.load(f)

    # 1. Get the reliable input height
    input_height = data_from_cli.get('input_height_cm', np.nan)

    # 2. Force all unreliable measurements to NaN to compel the imputer to predict them
    measurement_data: Dict[str, Any] = {
        'TotalHeight': input_height,
        # Force these to NaN to use the imputation model's estimate:
        'ShoulderWidth': np.nan,
        'ChestWidth': np.nan,
        'Waist': np.nan,
        'Hips': np.nan,
        # Add other features like 'Age' or 'Weight' here if you have them from another source
    }

    # 3. Create DataFrame row
    input_row_for_df = {k: measurement_data.get(k, np.nan) for k in IMPUTER_FEATURES}
    input_df = pd.DataFrame([input_row_for_df])

except Exception as e:
    sys.stderr.write(f"Error loading input data: {e}\n")
    print(json.dumps({"status": "error", "message": f"Input load error: {e}"}))
    sys.exit(1)

timings["step_1_setup_s"] = round((time.perf_counter() - t0), 3)

# ---------------------------
# 2. IMPUTATION LOGIC (Unchanged)
# ---------------------------
t0 = time.perf_counter()

try:
    imputer = joblib.load("final_imputer.pkl")

    # Align columns to the exact order expected by the imputer
    input_df_aligned = input_df[IMPUTER_FEATURES]

    # Impute missing values
    new_data_imputed_array = imputer.transform(input_df_aligned)

    # Convert to dictionary
    imputed_series = pd.Series(
        data=new_data_imputed_array[0],
        index=IMPUTER_FEATURES
    )

    final_measurements = imputed_series.to_dict()

except Exception as e:
    sys.stderr.write(f"Error during imputation or model loading: {e}\n")
    print(json.dumps({"status": "error", "message": f"Imputation error: {e}"}))
    sys.exit(1)

timings["step_2_impute_s"] = round((time.perf_counter() - t0), 3)

# ---------------------------
# 3. CLOTHING SIZE LOGIC
# ---------------------------
t0 = time.perf_counter()

# Use the imputed ChestWidth for sizing
chest_width = final_measurements.get('ChestWidth', 0)

# NOTE: Using your original thresholds (115, 105, etc.) but applied to the IMPUTED ChestWidth.
# This logic strongly suggests these thresholds were meant for a CIRCUMFERENCE measurement,
# not a WIDTH measurement. If ChestWidth is e.g. 45 cm, this will always return XS.
# Assuming your Imputer model uses 'ChestWidth' as a proxy for the required sizing dimension:
if chest_width > 115:
    recommended_size = "XL"
elif chest_width > 105:
    recommended_size = "L"
elif chest_width > 95:
    recommended_size = "M"
elif chest_width > 85:
    recommended_size = "S"
else:
    recommended_size = "XS"

timings["step_3_sizing_s"] = round((time.perf_counter() - t0), 3)

# ---------------------------
# 4. FINAL OUTPUT (Direct to STDOUT)
# ---------------------------
t_all_end = time.perf_counter()
timings["total_runtime_s"] = round((t_all_end - t_all), 3)

# Convert seconds to milliseconds for C# parsing (using _ms suffix)
runtime_ms = {k.replace('_s', '_ms'): round(v * 1000, 1) for k, v in timings.items()}

# Format the final output structure
final_results_for_unity = {
    "status": "success",
    "file_saved": "N/A (Output to stdout)",
    "path": input_json_path,

    # Core Measurement Data (These are the IMPUTED/Predicted values)
    'TotalHeight': final_measurements.get('TotalHeight', 0.0),
    'ShoulderWidth': final_measurements.get('ShoulderWidth', 0.0),
    'ChestWidth': final_measurements.get('ChestWidth', 0.0),
    'Waist': final_measurements.get('Waist', 0.0),
    'Hips': final_measurements.get('Hips', 0.0),
    'recommended_size': recommended_size,

    # Runtime Data
    'runtime_ms': runtime_ms
}

# The ONLY thing printed to stdout must be this:
print(json.dumps(final_results_for_unity, default=default_converter))