import sys
import json
import joblib
import pandas as pd
import numpy as np
import os
import traceback
import time  # <-- NEW: Added for timing
from typing import Dict, Any, Union, Tuple

# Configuration
TRANSFORMER_PATH = "transformer.pkl"
IMPUTER_PATH = "final_imputer.pkl"
# Removed hardcoded OUTPUT_FILE_PATH, it will be determined by argv[6] in main()
# OUTPUT_FILE_PATH = "Data/final_output.json"

# The 14 features used for data alignment (INPUT and OUTPUT dictionary structure).
ALL_IMPUTER_FEATURES = [
    'Gender', 'Age', 'HeadCircumference', 'ShoulderWidth', 'ChestWidth', 'ChestFrontWidth',
    'Belly', 'Waist', 'Hips', 'ArmLength', 'ShoulderToWaist', 'WaistToKnee',
    'LegLength', 'TotalHeight'
]

# The features that are SCALED and IMPUTED.
TRANSFORMER_FEATURES = [
    'Age', 'HeadCircumference', 'ShoulderWidth', 'ChestWidth',
    'Belly', 'Waist', 'Hips', 'ArmLength', 'ShoulderToWaist', 'WaistToKnee',
    'LegLength', 'TotalHeight'
]

# Mapping from JSON-input keys to Imputer's feature names (in cm).
MEASUREMENT_MAPPING = {
    "shoulder_width_cm": "ShoulderWidth",
    "chest_cm": "ChestWidth",
    "chest_width_cm": "ChestFrontWidth",
    "waist_cm": "Waist",
    "hip_cm": "Hips",
    "inseam_cm": "LegLength",
    "input_height_cm": "TotalHeight",
}

# Create a display mapping for debug messages
DEBUG_DISPLAY_NAMES = {
    "ChestWidth": "ChestCircumference",
    "ChestFrontWidth": "ChestFrontWidth",
    "ShoulderWidth": "ShoulderWidth",
    "Waist": "Waist",
    "Hips": "Hips",
    "LegLength": "LegLength",
    "TotalHeight": "TotalHeight",
    "Age": "Age",
    "Gender": "Gender",
    "HeadCircumference": "HeadCircumference",
    "ShoulderToWaist": "ShoulderToWaist",
    "WaistToKnee": "WaistToKnee",
    "ArmLength": "ArmLength",
}

# Features that should strictly use user input if provided (Static)
STATIC_USER_FEATURES = [
    "Age", "ShoulderWidth", "ChestWidth", "ChestFrontWidth", "Waist", "Hips", "TotalHeight", "Gender"
]

STATIC_GENDER_VALUE = 0.0  # Default to Female

# Size chart column mapping
# Maps the size chart column names to simpler names for use in the sizing logic.
SIZE_CHART_COLUMN_MAPPING = {
    'SIZE': 'SIZE',
    # Maps CHES(C) to a simplified name
    'CHEST (C)': 'CHART_CHEST_WIDTH',
    # Maps BODY LENGTH (BL) to a simplified name
    'BODY LENGTH (BL)': 'CHART_BODY_LENGTH',
    # Maps SLEEVE LENGTH (SL)': 'CHART_SLEEVE LENGTH'
    'SLEEVE LENGTH (SL)': 'CHART_SLEEVE_LENGTH'
}

# Define the order of sizes for comparison, from smallest to largest
SIZE_ORDER = ['S', 'M', 'L', 'XL', 'XXL']
SIZE_TO_INT = {size: i + 1 for i, size in enumerate(SIZE_ORDER)}
INT_TO_SIZE = {i + 1: size for i, size in enumerate(SIZE_ORDER)}


# Sizing:
# HARD CHECK:
#   A size is immediately 'disqualified' if the user's
#   ChestFrontWidth, ShoulderToWaist, or ArmLength is too big for that size
#   even when allowing a tolerance (standard deviation).
# BEST FIT:
#   For all sizes that pass the hard check, there is calculated a 'Total Mismatch Score'.
#   This score measures how far the user's three key measurements are from the chart's ideal numbers.
# RECOMMENDED SIZE:
#   The size with the 'lowest Mismatch Score' is chosen as the best fit.
# TOTAL MISMATCH SCORE:
#   The score is the square root of the sum of the squared differences across the 3 key features.
#   A small score means the body dimensions are very close to the chart's dimensions for that size.
# Total Mismatch Score = √((User's ChestWidth - Chart ChestWidth)² + (User's ShoulderToWaist - Chart ShoulderToWaist)² + (User's ArmLength  - Chart ArmLength)²)

def get_recommended_size(measurements: Dict[str, float], size_chart: pd.DataFrame) -> Tuple[str, str, str]:
    # IMPORTANT: These are the features used for the distance calculation.
    REQUIRED = ['ChestFrontWidth', 'ShoulderToWaist', 'ArmLength']

    # The standard deviation calculated in cm for each measurement to determine the size (standard deviation calculated from the Kaggle dataset)
    # if a user's measurement exceeds the chart value plus this tolerance, the size is 'disqualified' as too small.
    TOLERANCES = {
        'ChestFrontWidth': 13.50,
        'ShoulderToWaist': 13.66,
        'ArmLength': 13.66
    }

    # Check for missing data
    for col in REQUIRED:
        val = measurements.get(col)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            # Return error for all three required outputs
            return "Error", f"Missing measurement: {col}", "Error"

    # Map imputer features -> size chart columns
    CHART_COL_MAP = {
        'ChestFrontWidth': 'CHART_CHEST_WIDTH',
        'ShoulderToWaist': 'CHART_BODY_LENGTH',
        'ArmLength': 'CHART_SLEEVE_LENGTH'
    }

    # Make sure size ordering works
    size_chart = size_chart.copy()
    size_chart['size_int'] = size_chart['SIZE'].map(SIZE_TO_INT)
    possible_sizes = size_chart

    distances = []  # (size_str, size_int, distance_value)

    # Compute distance for each size in chart
    for _, row in possible_sizes.iterrows():
        dist = 0.0
        is_disqualified = False

        for m_col, sc_col in CHART_COL_MAP.items():
            user_val = measurements[m_col]
            chart_val = row[sc_col]

            # HARD LIMIT CHECK: DISQUALIFY SIZES THAT ARE TOO SMALL
            # If the user's measurement is LARGER than the chart measurement + tolerance, the size is disqualified.
            if m_col in TOLERANCES:
                tolerance = TOLERANCES[m_col]
                if user_val > chart_val + tolerance:
                    # Set the distance to infinity (disqualified)
                    dist = np.inf
                    is_disqualified = True
                    break  # Move to the next size

            # Standard (unweighted) Euclidean distance
            dist += (user_val - chart_val) ** 2

        # If the size was not disqualified
        if not is_disqualified:
            dist = np.sqrt(dist)

        distances.append((row['SIZE'], row['size_int'], dist))

    # Find smallest distance
    distances_sorted = sorted(distances, key=lambda x: x[2])
    best_size, best_int, best_dist = distances_sorted[0]

    # Initialize the new simple info string
    simple_additional_info = ""

    # HANDLING EXTREME WARNING (IF ALL SIZES ARE DISQUALIFIED)
    if best_dist == np.inf:
        warning_details = []

        # Since ALL sizes were disqualified, it means the user's measurements
        # exceeded the largest size ('XXL') + tolerance.
        # We check against the largest size ('XXL') to detail which measurements failed.

        # Safely get the largest size data
        largest_size_row = size_chart[size_chart['SIZE'] == 'XXL']
        largest_size_row = largest_size_row.iloc[0]

        for user_feature, chart_column in CHART_COL_MAP.items():
            if user_feature in TOLERANCES:
                user_val = measurements.get(user_feature)
                chart_val = largest_size_row[chart_column]
                tolerance = TOLERANCES[user_feature]

                # Check which measurements exceed XXL + tolerance
                if user_val > chart_val + tolerance:
                    display_name = DEBUG_DISPLAY_NAMES.get(user_feature, user_feature)
                    warning_details.append(
                        f"{display_name} ({user_val:.1f}cm vs. {chart_val:.1f}cm chart + {tolerance:.2f}cm tolerance)")

        details_str = " & ".join(warning_details)

        # Simple info for error state
        simple_additional_info = "WARNING: No standard size fits. Your measurements are too large."

        # Return a warning since no size fit
        return (
            f"ERROR: No size fits. (WARNING: Extreme tight fit - Your measurements are too large. Clothing is likely too small. Measured against largest size 'XXL': {details_str}.)",
            simple_additional_info,
            "ERROR")

    # Sizing message

    # Find nearest neighbours (for fit suggestion)
    second_best_size, second_best_int, second_best_dist = (None, None, np.inf)
    if len(distances_sorted) > 1:
        # First valid size after the winner (the second-best distance)
        for size, size_int, dist in distances_sorted[1:]:
            if dist != np.inf:
                second_best_size, second_best_int, second_best_dist = size, size_int, dist
                break

    # Determine which size should be used for DETAILED COMPARISON
    comparison_size = second_best_size  # Use the valid second-best, if found

    # If the best size is the largest ('XXL') AND there is no other valid size,
    # we use the next smallest size ('XL') for the comparison details.
    if best_size == 'XXL' and comparison_size is None:
        best_int = SIZE_TO_INT.get(best_size)
        if best_int is not None and best_int > 1:
            comparison_size = INT_TO_SIZE.get(best_int - 1)

    # If comparison size is still None, use the next size up or down based on index
    if comparison_size is None:
        best_int = SIZE_TO_INT.get(best_size)
        if best_int is not None:
            # Try next size up
            next_int = best_int + 1
            if next_int in INT_TO_SIZE and INT_TO_SIZE.get(next_int) in size_chart['SIZE'].values:
                comparison_size = INT_TO_SIZE.get(next_int)
            # Try next size down
            elif best_int > 1:
                prev_int = best_int - 1
                if prev_int in INT_TO_SIZE and INT_TO_SIZE.get(prev_int) in size_chart['SIZE'].values:
                    comparison_size = INT_TO_SIZE.get(prev_int)

    # Initialize message parts
    recommended_size = best_size
    tight_loose_suffix = ""
    comparison_details_line = ""
    suggestion_suffix = ""

    # Tight/Loose suggestion (Only if second_best_size is valid/non-inf)
    if second_best_size is not None and second_best_dist != np.inf:
        is_between_sizes = False
        second_best_row = size_chart[size_chart['SIZE'] == second_best_size].iloc[0]

        # Check if it's close enough (the soft boundary)
        for user_feature, chart_column in CHART_COL_MAP.items():
            if user_feature in TOLERANCES:
                user_val = measurements.get(user_feature)
                chart_val = second_best_row[chart_column]
                tolerance = TOLERANCES[user_feature]

                if abs(user_val - chart_val) < tolerance:
                    is_between_sizes = True
                    break

        if is_between_sizes:
            if second_best_int < best_int:
                # Loose fit
                tight_loose_suffix = " (Can have bit loose fit)"
                # This is the desired simple message part
                simple_additional_info = f"For a tighter fit, consider choosing size {second_best_size}."
            else:
                # Tight fit
                tight_loose_suffix = " (Can be a bit of a tight fit)"
                # This is the desired simple message part
                simple_additional_info = f"For a looser fit, consider choosing size {second_best_size}."

    if not simple_additional_info:
        simple_additional_info = f"Size {best_size} is the best fit."

    # Calculate and add cm distance to the comparison size
    comparison_details = []

    if comparison_size is not None:
        comparison_row = size_chart[size_chart['SIZE'] == comparison_size].iloc[0]

        for user_feature, chart_column in CHART_COL_MAP.items():
            user_val = measurements.get(user_feature)
            chart_val_next = comparison_row[chart_column]

            # Define the simplified display name for the comparison output
            display_map_for_comparison = {
                'ChestFrontWidth': 'Chest Width',
                'ShoulderToWaist': 'Torso Length',
                'ArmLength': 'Sleeve Length'
            }
            # Use the simplified name for the final output
            display_name = display_map_for_comparison.get(user_feature, user_feature)

            # Calculate the difference: User - Chart (Comparison Size)
            diff_cm = user_val - chart_val_next

            # Determine the symbol based on the difference:
            comparison_symbol = ">" if diff_cm >= 0 else "<"

            # Use absolute value for the distance, formatted to two decimal places
            abs_diff = abs(diff_cm)

            comparison_details.append(
                f"{display_name}: {comparison_symbol} {abs_diff:.2f} cm from {comparison_size}")

        # Final formatting
        if comparison_details:
            details_joined = "\n\n".join(comparison_details)

            # Structure the final output part:
            comparison_details_line = (
                f"\n\n<u>Compared with the next best size {comparison_size}:</u>\n\n"
                f"{details_joined}"
            )

    # Final composition of the message
    recommended_size_for_output = best_size

    # Re-assemble the message parts for the final output string:
    final_message_parts = [f"Recommended: {best_size}"]

    if tight_loose_suffix:
        final_message_parts.append(f"{tight_loose_suffix}")

    if comparison_details_line:
        # The comparison_details_line already contains the full "\n\nCompared with..." string.
        final_message_parts.append(comparison_details_line)

    # Join the parts to form the final string printed to output
    # NOTE: Join with "\n" to match the expected format for splitting in C#
    msg = "\n".join(final_message_parts)

    # We now return three values
    return msg, simple_additional_info, recommended_size_for_output


def main():
    # --- NEW: Timing variables ---
    start_time = time.time()
    timing_data = {
        "step_1_setup_ms": 0.0,
        "step_2_impute_ms": 0.0,
        "step_3_sizing_ms": 0.0,
        "step_4_export_ms": 0.0,
        "total_runtime_ms": 0.0
    }
    # ---------------------------

    # Check for basic arguments
    # Arguments expected: [1] input_json_path, [2] size_chart_csv_path, [3] height, [4] age, [5] gender, [6] run_data_folder_path
    if len(sys.argv) < 7:
        error_output = {"status": "error",
                        "recommended_size": "Argument Error: Missing required arguments. Expected: input_json_path, size_chart_csv_path, height, age, gender, **run_data_folder_path**",
                        # Include the new field in error output
                        "simple_additional_info": "System error: Missing arguments.",
                        "runtime_ms": timing_data}  # <-- Include timing in error
        print(json.dumps(error_output))
        sys.exit(1)

    input_json_path = sys.argv[1]
    size_chart_csv_path = sys.argv[2]
    # The new argument from Unity (C#) containing the 'Participant xx' folder path
    run_data_folder_path = sys.argv[6]

    # --- NEW: Define the output file path dynamically ---
    OUTPUT_FILE_NAME = "final_output.json"
    OUTPUT_FILE_PATH = os.path.join(run_data_folder_path, OUTPUT_FILE_NAME)
    # ---------------------------------------------------

    # Read command line arguments from Unity

    cmd_line_height = None
    if len(sys.argv) > 3:
        try:
            val = sys.argv[3]
            if val and val.lower() != "null" and val != "":
                cmd_line_height = float(val)
        except ValueError:
            pass

    cmd_line_age = None
    if len(sys.argv) > 4:
        try:
            val = sys.argv[4]
            if val and val.lower() != "null" and val != "":
                cmd_line_age = float(val)
        except ValueError:
            pass

    cmd_line_gender = None
    if len(sys.argv) > 5:
        try:
            val = sys.argv[5]
            if val and val.lower() != "null" and val != "":
                # C# passes 1.0 for Male, 0.0 for Female, 2.0 for Non-binary
                cmd_line_gender = float(val)
        except ValueError:
            pass

    # Helper function for quick error output
    def quick_error_exit(e: Exception, step_name: str, raw_traceback: str = ""):
        # Calculate total time before exiting
        timing_data['total_runtime_ms'] = (time.time() - start_time) * 1000.0

        error_output = {
            "status": "error",
            "debug_message": f"Error in {step_name}",
            "recommended_size": f"{step_name} Error: {type(e).__name__}: {e}",
            "scaled_measurements_json": "",
            "final_measurements_json": raw_traceback,
            # Include the new field in error output
            "simple_additional_info": f"System error during {step_name}.",
            "runtime_ms": timing_data  # <-- Include timing in error
        }
        print(json.dumps(error_output))
        sys.exit(1)

    # Loading and error handling
    size_chart = pd.DataFrame()
    imputer = None
    transformer = None
    gender_value = STATIC_GENDER_VALUE  # Default to 0.0 (Female)
    debug_message = ""
    sizing_log = ""
    # Initialize the new variable
    simple_additional_info = ""

    try:
        # Step 1: Setup and Loading (Start)
        if not os.path.exists(IMPUTER_PATH):
            raise FileNotFoundError(f"Imputer file not found: {IMPUTER_PATH}")
        if not os.path.exists(TRANSFORMER_PATH):
            raise FileNotFoundError(f"Transformer file not found: {TRANSFORMER_PATH}")

        with open(input_json_path, 'r', encoding='utf-8') as f:
            script1_data = json.load(f)

        # Check for Gender in the JSON data (fallback/initial check)
        json_gender = script1_data.get('qa', {}).get('normalization', {}).get('gender', None)
        if json_gender:
            if json_gender.lower() == 'male':
                gender_value = 1.0
            elif json_gender.lower() == 'female':
                gender_value = 0.0

        imputer = joblib.load(IMPUTER_PATH)
        transformer = joblib.load(TRANSFORMER_PATH)

        size_chart = pd.read_csv(size_chart_csv_path, sep=',', encoding='utf-8')
        if 'SIZE' not in size_chart.columns:
            size_chart = pd.read_csv(size_chart_csv_path, sep=';', encoding='utf-8')

        # Rename size chart column to simpler names
        new_columns = {}
        for original_name, simple_name in SIZE_CHART_COLUMN_MAPPING.items():
            if original_name in size_chart.columns:
                new_columns[original_name] = simple_name

        # Apply the renaming
        size_chart.rename(columns=new_columns, inplace=True)

        # Ensure that the required simplified columns are present
        required_simple_cols = list(SIZE_CHART_COLUMN_MAPPING.values())

        # Step 1: Setup and Loading (End)
        timing_data['step_1_setup_ms'] = (time.time() - start_time) * 1000.0

    except Exception as e:
        quick_error_exit(e, "Setup and Loading", traceback.format_exc())

    # Prepare input for imputer
    # Initialize with NaNs, set Gender based on JSON
    imputer_input: Dict[str, Union[float, np.nan]] = {feature: np.nan for feature in ALL_IMPUTER_FEATURES}
    imputer_input['Gender'] = float(gender_value)  # Set based on JSON or default (0.0)

    # Parse user measurement input (from JSON)
    found_inputs = []
    for imputer_feature in ALL_IMPUTER_FEATURES:
        json_key = None
        for k, v in MEASUREMENT_MAPPING.items():
            if v == imputer_feature:
                json_key = k
                break

        if json_key:
            value = script1_data.get(json_key)
            if value is not None:
                try:
                    float_val = float(value)
                    imputer_input[imputer_feature] = float_val
                    # Use display name here
                    display_name = DEBUG_DISPLAY_NAMES.get(imputer_feature, imputer_feature)
                    found_inputs.append(f"{display_name}({json_key})")
                except (ValueError, TypeError):
                    pass

    # Inject Command Line Arguments (Overrides JSON if present)
    if cmd_line_height is not None:
        imputer_input['TotalHeight'] = cmd_line_height
        found_inputs.append(f"TotalHeight(UnityArg: {cmd_line_height})")

    if cmd_line_age is not None:
        imputer_input['Age'] = cmd_line_age
        found_inputs.append(f"Age(UnityArg: {cmd_line_age})")

    # Inject command line gender (Overrides JSON and default if present)
    if cmd_line_gender is not None:
        imputer_input['Gender'] = cmd_line_gender

        if cmd_line_gender == 1.0:
            gender_display = "Male"
        elif cmd_line_gender == 0.0:
            gender_display = "Female"
        elif cmd_line_gender == 2.0:
            gender_display = "Non-binary"
        else:
            gender_display = f"Unknown({cmd_line_gender})"

        found_inputs.append(f"Gender(UnityArg: {gender_display})")

    if found_inputs:
        debug_message += f"Inputs Found: {', '.join(found_inputs)}. "
    else:
        debug_message += "No matching inputs found. "

    input_df = pd.DataFrame([imputer_input])
    input_df_aligned = input_df[ALL_IMPUTER_FEATURES]

    # Transformation, Imputation, and Inverse-Transformation ---
    scaled_measurements_json_str = ""
    imputed_measurements_json_str = ""

    # Calculate time spent until here
    step_1_end_time = time.time()

    try:
        # Step 2: Imputation/Transformation (Start)
        # Prepare DataFrames for Transformer (features in TRANSFORMER_FEATURES)
        input_transform_df = input_df_aligned[TRANSFORMER_FEATURES].copy()

        # Forward Transformation: Scale the available data
        scaled_array = transformer.transform(input_transform_df)
        scaled_df = pd.DataFrame(scaled_array, columns=TRANSFORMER_FEATURES)

        # Imputation
        scaled_imputed_array = imputer.transform(scaled_df)
        scaled_imputed_measurements_df = pd.DataFrame(scaled_imputed_array, columns=TRANSFORMER_FEATURES)

        # Save the scaled measurements for debug
        scaled_measurements_dict = scaled_imputed_measurements_df.iloc[0].to_dict()
        scaled_measurements_json_str = json.dumps(scaled_measurements_dict)

        # De-scaling (Inverse Transformation)
        descaled_array = transformer.inverse_transform(scaled_imputed_measurements_df)
        imputed_df_transformed = pd.DataFrame(descaled_array, columns=TRANSFORMER_FEATURES)

        # Re-assemble the final imputed dataframe (start with the full input)
        imputed_df = input_df_aligned.copy()

        # Overwrite the imputable columns with the newly imputed values
        # This keeps non-imputed columns (like ChestFrontWidth) as their original input values
        for feature in TRANSFORMER_FEATURES:
            imputed_df[feature] = imputed_df_transformed[feature]

        # Overwrite imputed values with user input (Static values)
        original_row = input_df_aligned.iloc[0]

        overwritten_logs = []
        for feature in STATIC_USER_FEATURES:
            if feature in imputed_df.columns:
                user_value = original_row[feature]
                # If user provided a value (not NaN), use it.
                if not pd.isna(user_value):
                    imputed_df[feature] = user_value
                    overwritten_logs.append(feature)

        debug_message += f"Used Static Values for: {', '.join(overwritten_logs)}. "

        # Extract final dictionary (includes all 14 features)
        imputed_measurements = imputed_df[ALL_IMPUTER_FEATURES].iloc[0].to_dict()
        imputed_measurements = {k: float(v) for k, v in imputed_measurements.items() if not pd.isna(v)}

        # Step 2: Imputation/Transformation (End)
        timing_data['step_2_impute_ms'] = (time.time() - step_1_end_time) * 1000.0

    except Exception as e:
        quick_error_exit(e, "Transformation/Imputation", traceback.format_exc())

    # Calculate time spent until here
    step_2_end_time = time.time()

    # Sizing and output
    try:
        # Step 3: Sizing (Start)
        # NOTE: Now unpacks three return values
        recommended_text, simple_additional_info, sizing_log = get_recommended_size(imputed_measurements, size_chart)

        # Send the entire sentence to Unity
        recommended_size = recommended_text

        # Step 3: Sizing (End)
        timing_data['step_3_sizing_ms'] = (time.time() - step_2_end_time) * 1000.0

    except Exception as e:
        quick_error_exit(e, "Sizing Calculation", traceback.format_exc())

    # Calculate time spent until here
    step_3_end_time = time.time()

    # The debug message should not receive the recommended text
    debug_message += ""

    # Define the desired output order for the final_measurements_json
    ORDERED_OUTPUT_KEYS = [
        'Gender', 'Age', 'HeadCircumference', 'ShoulderWidth', 'ChestCircumference', 'ChestFrontWidth',
        'Belly', 'Waist', 'Hips', 'ArmLength', 'ShoulderToWaist', 'WaistToKnee',
        'LegLength', 'TotalHeight'
    ]

    # Rename 'ChestWidth' (circumference) to 'ChestCircumference'
    if 'ChestWidth' in imputed_measurements:
        # Use pop to rename the key and ensure only one version exists
        imputed_measurements['ChestCircumference'] = imputed_measurements.pop('ChestWidth')

    # Rebuild the dictionary in the correct order
    ordered_measurements = {}
    for key in ORDERED_OUTPUT_KEYS:
        if key in imputed_measurements:
            ordered_measurements[key] = imputed_measurements[key]

    # Use the newly ordered dictionary for the final output string
    imputed_measurements_json_str = json.dumps(ordered_measurements)

    # Calculate total time
    total_time_ms = (time.time() - start_time) * 1000.0
    timing_data['total_runtime_ms'] = total_time_ms
    timing_data['step_4_export_ms'] = (time.time() - step_3_end_time) * 1000.0  # Calculate export/finalizing time

    output_data = {
        "status": "success",
        "debug_message": debug_message,
        "recommended_size": recommended_size,  # Full detailed size string
        "simple_additional_info": simple_additional_info,  # NEW simple string
        "scaled_measurements_json": scaled_measurements_json_str,
        "final_measurements_json": imputed_measurements_json_str,
        "runtime_ms": timing_data  # <-- NEW: Include runtime data
    }

    final_json_output = json.dumps(output_data, indent=4)

    # We don't need to check for the directory since C# should have created it already.
    # But we ensure the full path is used.
    with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:
        f.write(final_json_output)

    # Print the final JSON to standard output for the C# script to capture
    print(json.dumps(output_data))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        tb = traceback.format_exc()
        # Calculate total time for crash exit
        total_time_ms = (
                                    time.time() - time.time()) * 1000.0  # Time since start is unknown, but use zero for consistency
        crash_timing_data = {
            "step_1_setup_ms": 0.0, "step_2_impute_ms": 0.0, "step_3_sizing_ms": 0.0, "step_4_export_ms": 0.0,
            "total_runtime_ms": total_time_ms
        }
        # Include the new field in error output
        print(json.dumps({"status": "error", "recommended_size": f"Crash: {str(e)}",
                          "simple_additional_info": "System error: Python script crashed.",
                          "final_measurements_json": tb,
                          "runtime_ms": crash_timing_data}))  # <-- Include timing in crash error
        sys.exit(1)