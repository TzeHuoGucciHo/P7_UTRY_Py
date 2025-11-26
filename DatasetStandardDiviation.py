import pandas as pd
import numpy as np

# Navnet på den uploadede fil er nu kun filnavnet
FILE_PATH = r"Mendeley Datasets/Body Measurements _ original_CSV.csv"

# Kolonner, vi skal bruge til toleranceberegning.
MEASUREMENT_COLUMNS = [
    'ChestWidth ',      # OBS: Mellemrum efter 'ChestWidth'
    'ShoulderToWaist ', # OBS: Mellemrum efter 'ShoulderToWaist'
    'ArmLength '        # OBS: Mellemrum efter 'ArmLength'
]

# Konverteringsfaktor: 1 inch = 2.54 cm
INCHES_TO_CM = 2.54


def calculate_std_devs(file_path: str):
    # 1. Indlæs data
    df = pd.read_csv(file_path)

    # 2. Rengøring og Udvalg
    # Vi udelukker rækker med manglende værdier (NaN) i de kolonner, vi vil analysere.
    df_clean = df.dropna(subset=MEASUREMENT_COLUMNS)

    # 3. Konvertering til cm
    # Multiplicer de relevante kolonner med konverteringsfaktoren
    for col in MEASUREMENT_COLUMNS:
        # Konverterer fra inches til cm (inches * 2.54)
        df_clean[col] = df_clean[col] * INCHES_TO_CM

    print(f"Data indlæst og konverteret til cm. Brugte {len(df_clean)} af {len(df)} rækker.")
    print("-" * 40)

    # 4. Beregn Standardafvigelse for hver kolonne
    std_devs = {}
    for col in MEASUREMENT_COLUMNS:
        # Beregn Standardafvigelse
        std_dev = df_clean[col].std()
        std_devs[col] = std_dev

    # 5. Udskriv resultater
    print("Standardafvigelser (σ) for measurements (i cm):")

    display_mapping = {
        'ChestWidth ': 'Brystmål (ChestWidth)',
        'ShoulderToWaist ': 'Kropslængde (ShoulderToWaist)',
        'ArmLength ': 'Ærmelængde (ArmLength)'
    }

    # Udskriv i tabelformat
    print("{:<30} {:>10}".format("Mål", "σ (cm)"))
    print("-" * 40)
    for col, sigma in std_devs.items():
        # Brug .get() for at håndtere tilfælde, hvor nøglen har et mellemrum
        display_name = display_mapping.get(col, col.strip())
        print("{:<30} {:>10.2f}".format(display_name, sigma))

if __name__ == "__main__":
    calculate_std_devs(FILE_PATH)