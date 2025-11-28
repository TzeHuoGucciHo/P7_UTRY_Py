import pandas as pd
import numpy as np

FILE_PATH = r"Mendeley Datasets/Body Measurements _ original_CSV.csv"

# Columns we need for tolerance calculation.
MEASUREMENT_COLUMNS = [
    'ChestWidth ',
    'ShoulderToWaist ',
    'ArmLength '
]

# Conversion factor: 1 inch = 2.54 cm
INCHES_TO_CM = 2.54


def calculate_std_devs(file_path: str):
    # Load data
    df = pd.read_csv(file_path)

    # Cleaning and selection
    # We exclude rows with missing values (NaN) in the columns we want to analyze.
    df_clean = df.dropna(subset=MEASUREMENT_COLUMNS)

    # Conversion to cm
    # Multiply the relevant columns by the conversion factor
    for col in MEASUREMENT_COLUMNS:
        # Converts from inches to cm (inches * 2.54)
        df_clean[col] = df_clean[col] * INCHES_TO_CM

    print(f"Data loaded and converted to cm. Used {len(df_clean)} out of {len(df)} rows.")

    # Calculate Standard Deviation for each column
    std_devs = {}
    for col in MEASUREMENT_COLUMNS:
        # Calculate Standard Deviation
        std_dev = df_clean[col].std()
        std_devs[col] = std_dev

    # Print results
    print("Standard Deviations (σ) for measurements (in cm):")

    display_mapping = {
        'ChestWidth ': 'Chest Measurement (ChestWidth)',
        'ShoulderToWaist ': 'Body Length (ShoulderToWaist)',
        'ArmLength ': 'Sleeve Length (ArmLength)'
    }

    # Print in table format
    print("{:<30} {:>10}".format("Measurement", "σ (cm)"))
    print("-" * 40)
    for col, sigma in std_devs.items():
        # Use .get() to handle cases where the key has a trailing space
        display_name = display_mapping.get(col, col.strip())
        print("{:<30} {:>10.2f}".format(display_name, sigma))

if __name__ == "__main__":
    calculate_std_devs(FILE_PATH)