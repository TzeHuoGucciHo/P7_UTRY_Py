import sys
import json
import joblib
import pandas as pd
import numpy as np
import os
import traceback
from typing import Dict, Any, Union, Tuple

# --- 1. CONFIGURATION ---

TRANSFORMER_PATH = "transformer.pkl"
IMPUTER_PATH = "final_imputer.pkl"
OUTPUT_FILE_PATH = "Data/final_output.json"

# The 14 features used for data alignment (INPUT and OUTPUT dictionary structure).
ALL_IMPUTER_FEATURES = [
    'Gender', 'Age', 'HeadCircumference', 'ShoulderWidth', 'ChestWidth', 'ChestFrontWidth',
    'Belly', 'Waist', 'Hips', 'ArmLength', 'ShoulderToWaist', 'WaistToKnee',
    'LegLength', 'TotalHeight'
]

# CRITICAL: This list defines the features that are SCALED and IMPUTED.
TRANSFORMER_FEATURES = [
    'Age', 'HeadCircumference', 'ShoulderWidth', 'ChestWidth',
    'Belly', 'Waist', 'Hips', 'ArmLength', 'ShoulderToWaist', 'WaistToKnee',
    'LegLength', 'TotalHeight'
]

# Mapping from JSON-input keys to Imputer's feature names (in CM).
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

# --- NEW: SIZE CHART COLUMN MAPPING ---
# Maps the size chart column names to simpler names for use in the sizing logic.
SIZE_CHART_COLUMN_MAPPING = {
    'SIZE': 'SIZE',
    # Maps CHES(C) to a simplified name
    'CHEST (C)': 'CHART_CHEST_WIDTH',
    # Maps BODY LENGTH (BL) to a simplified name
    'BODY LENGTH (BL)': 'CHART_BODY_LENGTH',
    # Maps SLEEVE LENGTH (SL)': 'CHART_SLEEVE_LENGTH'
    'SLEEVE LENGTH (SL)': 'CHART_SLEEVE_LENGTH'
}

# Define the order of sizes for comparison, from smallest to largest
SIZE_ORDER = ['S', 'M', 'L', 'XL', 'XXL']
SIZE_TO_INT = {size: i + 1 for i, size in enumerate(SIZE_ORDER)}
INT_TO_SIZE = {i + 1: size for i, size in enumerate(SIZE_ORDER)}


# --- 2. SIZING LOGIC (Opdateret) ---

def get_recommended_size(measurements: Dict[str, float], size_chart: pd.DataFrame):
    """
    Balanced sizing: Bruger standard (uvægtet) Euclidean afstand,
    men diskvalificerer størrelser, der er for små ifølge de specificerede tolerancer.
    Inkluderer detaljeret sammenligning med den næstbedste størrelse.
    """

    # VIGTIGT: Dette er de features, der bruges til afstandsberegningen.
    REQUIRED = ['ChestFrontWidth', 'ShoulderToWaist', 'ArmLength']

    # Standardafvigelsen beregnet i cm for hvert mål til at bestemme størrelsen (standardafvigelsen udregnet fra Kaggle datasæt)
    TOLERANCES = {
        'ChestFrontWidth': 13.50,
        'ShoulderToWaist': 13.66,
        'ArmLength': 13.66
    }

    # check for missing data
    for col in REQUIRED:
        val = measurements.get(col)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "Error", f"Missing measurement: {col}"

    # map imputer features -> size chart columns
    CHART_COL_MAP = {
        'ChestFrontWidth': 'CHART_CHEST_WIDTH',
        'ShoulderToWaist': 'CHART_BODY_LENGTH',
        'ArmLength': 'CHART_SLEEVE_LENGTH'
    }

    # make sure size ordering works
    size_chart = size_chart.copy()
    size_chart['size_int'] = size_chart['SIZE'].map(SIZE_TO_INT)
    possible_sizes = size_chart

    distances = []  # (size_str, size_int, distance_value)

    # compute distance for each size in chart
    for _, row in possible_sizes.iterrows():
        dist = 0.0
        is_disqualified = False

        for m_col, sc_col in CHART_COL_MAP.items():
            user_val = measurements[m_col]
            chart_val = row[sc_col]

            # --- 1. HÅRDT GRÆNSE-TJEK: DISKVALIFICER STØRRELSER, DER ER FOR SMÅ ---
            # Hvis brugerens mål er STØRRE end chart-målet + tolerance, diskvalificeres størrelsen.
            if m_col in TOLERANCES:
                tolerance = TOLERANCES[m_col]
                if user_val > chart_val + tolerance:
                    # Sæt afstanden til uendelig (diskvalificeret)
                    dist = np.inf
                    is_disqualified = True
                    break  # Gå videre til næste størrelse

            # Standard (uvægtet) Euclidean afstand
            dist += (user_val - chart_val) ** 2

        # Hvis størrelsen ikke blev diskvalificeret
        if not is_disqualified:
            dist = np.sqrt(dist)

        distances.append((row['SIZE'], row['size_int'], dist))

    # find smallest distance
    distances_sorted = sorted(distances, key=lambda x: x[2])
    best_size, best_int, best_dist = distances_sorted[0]

    # --- 2. HÅNDTERING AF EKSTREM ADVARSEL (HVIS ALLE STØRRELSER ER DISKVALIFICERET) ---
    if best_dist == np.inf:
        warning_details = []

        # Da alle størrelser er diskvalificeret, tjekker vi mod den mindste størrelse ('S')
        smallest_size_row = size_chart[size_chart['SIZE'] == 'S'].iloc[0]

        for user_feature, chart_column in CHART_COL_MAP.items():
            if user_feature in TOLERANCES:
                user_val = measurements.get(user_feature)
                chart_val = smallest_size_row[chart_column]
                tolerance = TOLERANCES[user_feature]

                # Tjek hvilke mål, der overskrider S + tolerance
                if user_val > chart_val + tolerance:
                    display_name = DEBUG_DISPLAY_NAMES.get(user_feature, user_feature)
                    warning_details.append(
                        f"{display_name} ({user_val:.1f}cm vs. {chart_val:.1f}cm chart + {tolerance:.2f}cm tolerance)")

        details_str = " & ".join(warning_details)

        # Returner en speciel advarsel, da ingen størrelse passede
        return f"ERROR: No size fits. (WARNING: Extreme tight fit - Your measurements are too large. Clothing is likely too small. Measured against smallest size 'S': {details_str}.)", ""

    # --- 3. SIZING MESSAGE OPBYGNING ---

    # 3a. find nearest neighbours (for fit suggestion)
    second_best_size, second_best_int, second_best_dist = (None, None, np.inf)
    if len(distances_sorted) > 1:
        # Første valide størrelse efter vinderen (den næstbedste afstand)
        for size, size_int, dist in distances_sorted[1:]:
            if dist != np.inf:
                second_best_size, second_best_int, second_best_dist = size, size_int, dist
                break

    # 3b. Bestem hvilken størrelse der skal bruges til DETALJERET SAMMENLIGNING
    comparison_size = second_best_size  # Brug den valide næstbedste, hvis den findes

    # Hvis den bedste størrelse er den største ('XXL') OG der ikke findes en anden valid størrelse,
    # bruger vi den næstmindste størrelse ('XL') til sammenligningsdetaljerne.
    if best_size == 'XXL' and comparison_size is None:
        best_int = SIZE_TO_INT.get(best_size)
        if best_int is not None and best_int > 1:
            comparison_size = INT_TO_SIZE.get(best_int - 1)

    # Initialiser beskeddele
    recommended_size = best_size
    msg_base = f"Recommended: {best_size}"
    tight_loose_suffix = ""
    comparison_details_line = ""
    suggestion_suffix = ""

    # --- TIGHT/LOOSE SUGGESTION (KUN HVIS second_best_size ER GYLDIG/NON-INF) ---
    if second_best_size is not None and second_best_dist != np.inf:
        is_between_sizes = False
        second_best_row = size_chart[size_chart['SIZE'] == second_best_size].iloc[0]

        # Tjek om den er tæt nok på (den bløde grænse)
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
                tight_loose_suffix = " (can have a bit loose fit)"
                suggestion_suffix = f" - choose {second_best_size} for a tighter fit"
            else:
                # Tight fit (Brugerens ønskede formulering)
                tight_loose_suffix = " (can be a bit of a tight fit)"
                suggestion_suffix = f" - choose {second_best_size} for a looser fit"

    # --- BEREGN OG TILFØJ CM AFSTAND TIL SAMMENLIGNINGSSTØRRELSEN ---
    comparison_details = []

    if comparison_size is not None:
        comparison_row = size_chart[size_chart['SIZE'] == comparison_size].iloc[0]

        for user_feature, chart_column in CHART_COL_MAP.items():
            user_val = measurements.get(user_feature)
            chart_val_next = comparison_row[chart_column]
            display_name = DEBUG_DISPLAY_NAMES.get(user_feature, user_feature)

            # Beregn forskellen: Bruger - Chart (Sammenligningsstørrelse)
            diff_cm = user_val - chart_val_next

            # Sæt præposition baseret på om brugeren er større (+) eller mindre (-) end målet
            if diff_cm > 0:
                preposition_txt = "larger than"
            else:
                preposition_txt = "smaller than"

            # Sætter altid absolut værdi for læsbarhed
            comparison_details.append(
                f"{display_name}: {abs(diff_cm):.2f} cm {preposition_txt} {comparison_size}")

        # Føj de nye detaljer til output strengen
        if comparison_details:
            # Sætter detaljerne på en ny linje
            comparison_details_line = f"\n(Compared to the next best size ({comparison_size}): {'; '.join(comparison_details)})"

    # 4. Endelig sammensætning af beskeden
    # Vi samler nu beskeden i den ønskede rækkefølge: base + tight/loose + detaljer + suggestion
    msg = msg_base + tight_loose_suffix + comparison_details_line + suggestion_suffix

    return msg, recommended_size


def main():
    # Removed t_start = time.perf_counter()

    # Check for basic arguments (Script path, Input JSON, Size Chart CSV)
    if len(sys.argv) < 3:
        error_output = {"status": "error",
                        "recommended_size": "Argument Error: Missing input_json_path (sys.argv[1]) and size_chart_csv_path (sys.argv[2])"}
        print(json.dumps(error_output))
        sys.exit(1)

    input_json_path = sys.argv[1]
    size_chart_csv_path = sys.argv[2]

    # --- Read Command Line Arguments from Unity/C# ---

    # Check for User Height passed from Unity as 3rd argument (sys.argv[3])
    cmd_line_height = None
    if len(sys.argv) > 3:
        try:
            val = sys.argv[3]
            if val and val.lower() != "null" and val != "":
                cmd_line_height = float(val)
        except ValueError:
            pass

    # Check for User Age passed from Unity as 4th argument (sys.argv[4])
    cmd_line_age = None
    if len(sys.argv) > 4:
        try:
            val = sys.argv[4]
            if val and val.lower() != "null" and val != "":
                cmd_line_age = float(val)
        except ValueError:
            pass

    # Check for User Gender passed from Unity as 5th argument (sys.argv[5])
    cmd_line_gender = None
    if len(sys.argv) > 5:
        try:
            val = sys.argv[5]
            if val and val.lower() != "null" and val != "":
                # C# passes 1.0 for Male, 0.0 for Female, 2.0 for Non-binary
                cmd_line_gender = float(val)
        except ValueError:
            pass

    # --- Helper function for quick error output ---
    def quick_error_exit(e: Exception, step_name: str, raw_traceback: str = ""):
        error_output = {
            "status": "error",
            "debug_message": f"Error in {step_name}",
            "recommended_size": f"{step_name} Error: {type(e).__name__}: {e}",
            "scaled_measurements_json": "",
            "final_measurements_json": raw_traceback,
            # Removed runtime_ms / runtime_s from the error output structure
        }
        print(json.dumps(error_output))
        sys.exit(1)

    # --- LOADING AND ERROR HANDLING ---
    # Removed t_setup_start = time.perf_counter()
    size_chart = pd.DataFrame()
    imputer = None
    transformer = None
    gender_value = STATIC_GENDER_VALUE  # Default to 0.0 (Female)
    debug_message = ""
    sizing_log = ""

    try:
        if not os.path.exists(IMPUTER_PATH):
            raise FileNotFoundError(f"Imputer file not found: {IMPUTER_PATH}")
        if not os.path.exists(TRANSFORMER_PATH):
            raise FileNotFoundError(f"Transformer file not found: {TRANSFORMER_PATH}")

        with open(input_json_path, 'r', encoding='utf-8') as f:
            script1_data = json.load(f)

        # 1. Check for Gender in the JSON data (fallback/initial check)
        json_gender = script1_data.get('qa', {}).get('normalization', {}).get('gender', None)
        if json_gender:
            if json_gender.lower() == 'male':
                gender_value = 1.0
            elif json_gender.lower() == 'female':
                gender_value = 0.0

        imputer = joblib.load(IMPUTER_PATH)
        transformer = joblib.load(TRANSFORMER_PATH)

        try:
            size_chart = pd.read_csv(size_chart_csv_path, sep=',', encoding='utf-8')
            if 'SIZE' not in size_chart.columns:
                size_chart = pd.read_csv(size_chart_csv_path, sep=';', encoding='utf-8')

            # --- RENAME SIZE CHART COLUMNS TO SIMPLER NAMES ---
            # This makes the sizing logic cleaner and less dependent on specific CSV headers.
            new_columns = {}
            for original_name, simple_name in SIZE_CHART_COLUMN_MAPPING.items():
                if original_name in size_chart.columns:
                    new_columns[original_name] = simple_name

            # Apply the renaming
            size_chart.rename(columns=new_columns, inplace=True)

            # CRITICAL CHECK: Ensure the required simplified columns are present
            required_simple_cols = list(SIZE_CHART_COLUMN_MAPPING.values())
            if not all(col in size_chart.columns for col in required_simple_cols):
                missing = [col for col in required_simple_cols if col not in size_chart.columns]
                raise ValueError(
                    f"Size chart is missing required columns after loading/renaming. Missing simplified columns: {missing}")

        except Exception as e:
            raise Exception(f"Error loading and renaming size chart: {e}")

    except Exception as e:
        quick_error_exit(e, "Setup and Loading", traceback.format_exc())

    # Removed t_setup_end = time.perf_counter()

    # --- PREPARE INPUT FOR IMPUTER ---
    # 1. Initialize with NaNs, set Gender based on JSON/Default
    imputer_input: Dict[str, Union[float, np.nan]] = {feature: np.nan for feature in ALL_IMPUTER_FEATURES}
    imputer_input['Gender'] = float(gender_value)  # Set based on JSON or default (0.0)

    # 2. Parse User Measurement Input (from JSON)
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

    # 3. Inject Command Line Arguments (Overrides JSON if present)
    if cmd_line_height is not None:
        imputer_input['TotalHeight'] = cmd_line_height
        found_inputs.append(f"TotalHeight(UnityArg: {cmd_line_height})")

    if cmd_line_age is not None:
        imputer_input['Age'] = cmd_line_age
        found_inputs.append(f"Age(UnityArg: {cmd_line_age})")

    # Inject Command Line Gender (Overrides JSON and default if present)
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

    # --- 3. TRANSFORMATION, IMPUTATION, AND INVERSE-TRANSFORMATION ---
    # Removed t_impute_start = time.perf_counter()
    scaled_measurements_json_str = ""
    imputed_measurements_json_str = ""

    try:
        # 1. Prepare DataFrames for Transformer (features in TRANSFORMER_FEATURES)
        input_transform_df = input_df_aligned[TRANSFORMER_FEATURES].copy()

        # 2. Forward Transformation: Scale the available data
        scaled_array = transformer.transform(input_transform_df)
        scaled_df = pd.DataFrame(scaled_array, columns=TRANSFORMER_FEATURES)

        # 3. Imputation
        scaled_imputed_array = imputer.transform(scaled_df)
        scaled_imputed_measurements_df = pd.DataFrame(scaled_imputed_array, columns=TRANSFORMER_FEATURES)

        # 4. Save the scaled measurements for debug
        scaled_measurements_dict = scaled_imputed_measurements_df.iloc[0].to_dict()
        scaled_measurements_json_str = json.dumps(scaled_measurements_dict)

        # 5. De-scaling (Inverse Transformation)
        descaled_array = transformer.inverse_transform(scaled_imputed_measurements_df)
        imputed_df_transformed = pd.DataFrame(descaled_array, columns=TRANSFORMER_FEATURES)

        # 6. Re-assemble the final imputed dataframe (start with the full input)
        imputed_df = input_df_aligned.copy()

        # 7. Overwrite the imputable columns with the newly imputed values
        # This keeps non-imputed columns (like ChestFrontWidth) as their original input values
        for feature in TRANSFORMER_FEATURES:
            imputed_df[feature] = imputed_df_transformed[feature]

        # --------------------------------------------------------------------
        # 8. OVERWRITE IMPUTED VALUES WITH USER INPUT (STATIC VALUES)
        # --------------------------------------------------------------------
        original_row = input_df_aligned.iloc[0]

        overwritten_logs = []
        for feature in STATIC_USER_FEATURES:
            if feature in imputed_df.columns:
                user_value = original_row[feature]
                # If user provided a value (not NaN), use it.
                if not pd.isna(user_value):
                    imputed_df[feature] = user_value
                    overwritten_logs.append(feature)

        if overwritten_logs:
            debug_message += f"Used Static Values for: {', '.join(overwritten_logs)}. "
        else:
            debug_message += "No Static Values applied. "

        # 9. Extract final dictionary (includes all 14 features)
        imputed_measurements = imputed_df[ALL_IMPUTER_FEATURES].iloc[0].to_dict()
        imputed_measurements = {k: float(v) for k, v in imputed_measurements.items() if not pd.isna(v)}

    except Exception as e:
        quick_error_exit(e, "Transformation/Imputation", traceback.format_exc())

    # Removed t_impute_end = time.perf_counter()

    # --- 4. SIZING & OUTPUT ---
    recommended_text, sizing_log = get_recommended_size(imputed_measurements, size_chart)

    # send hele sætningen til Unity
    recommended_size = recommended_text

    # debug må ikke få recommended text
    debug_message += ""

    # Define the desired output order for the final_measurements_json
    ORDERED_OUTPUT_KEYS = [
        'Gender', 'Age', 'HeadCircumference', 'ShoulderWidth', 'ChestCircumference', 'ChestFrontWidth',
        'Belly', 'Waist', 'Hips', 'ArmLength', 'ShoulderToWaist', 'WaistToKnee',
        'LegLength', 'TotalHeight'
    ]

    # Step 1: Rename 'ChestWidth' (circumference) to 'ChestCircumference'
    if 'ChestWidth' in imputed_measurements:
        # Use pop to rename the key and ensure only one version exists
        imputed_measurements['ChestCircumference'] = imputed_measurements.pop('ChestWidth')

    # Step 2: Rebuild the dictionary in the correct order
    ordered_measurements = {}
    for key in ORDERED_OUTPUT_KEYS:
        if key in imputed_measurements:
            ordered_measurements[key] = imputed_measurements[key]

    # Use the newly ordered dictionary for the final output string
    imputed_measurements_json_str = json.dumps(ordered_measurements)
    # Removed t_end = time.perf_counter()

    # Removed runtime_data calculation block

    output_data = {
        "status": "success",
        "debug_message": debug_message,
        "recommended_size": recommended_size,
        "scaled_measurements_json": scaled_measurements_json_str,
        "final_measurements_json": imputed_measurements_json_str,
    }

    final_json_output = json.dumps(output_data, indent=4)

    # File Saving
    try:
        output_dir = os.path.dirname(OUTPUT_FILE_PATH)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:
            f.write(final_json_output)
    except Exception:
        pass

    # Print the final JSON to standard output for the C# script to capture
    print(json.dumps(output_data))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(json.dumps({"status": "error", "recommended_size": f"Crash: {str(e)}"}))
        sys.exit(1)