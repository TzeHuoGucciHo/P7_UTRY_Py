import pandas as pd
import numpy as np

# Navnet på den uploadede fil er nu kun filnavnet
FILE_PATH = r"C:\Uni\MED7\Semester project\P7_UTRY_Py\Mendeley Datasets\Body Measurements _ original_CSV.csv"

# Kolonner, vi skal bruge til toleranceberegning. De skal matche filen PRÆCIST.
MEASUREMENT_COLUMNS = [
    'ChestWidth ',      # OBS: Mellemrum efter 'ChestWidth'
    'ShoulderToWaist ', # OBS: Mellemrum efter 'ShoulderToWaist'
    'ArmLength '        # OBS: Mellemrum efter 'ArmLength'
]


def calculate_std_devs(file_path: str):
    """
    Indlæser kropsmålingsdata og beregner standardafvigelsen (sigma) for nøglemålinger.
    """
    try:
        # 1. Indlæs data
        df = pd.read_csv(file_path)

        # 2. Rengøring og Udvalg
        # Vi udelukker rækker med manglende værdier (NaN) i de kolonner, vi vil analysere.
        df_clean = df.dropna(subset=MEASUREMENT_COLUMNS)

        if df_clean.empty:
            print("Fejl: Datasættet er tomt efter fjernelse af manglende værdier.")
            return

        print(f"✅ Data indlæst succesfuldt. Brugte {len(df_clean)} af {len(df)} rækker.")
        print("-" * 40)

        # 3. Beregn Standardafvigelse for hver kolonne
        std_devs = {}
        for col in MEASUREMENT_COLUMNS:
            # Beregn Standardafvigelse (sigma)
            std_dev = df_clean[col].std()
            std_devs[col] = std_dev

        # 4. Udskriv resultater
        print("📊 Standardafvigelser (σ) for nøglemålinger (i cm):")

        display_mapping = {
            'ChestWidth': 'Brystmål (Chest Circumference)',
            'ShoulderToWaist': 'Kropslængde (ShoulderToWaist)',
            'ArmLength': 'Ærmelængde (ArmLength)'
        }

        # Udskriv i tabelformat
        print("{:<30} {:>10}".format("Mål", "σ (cm)"))
        print("-" * 40)
        for col, sigma in std_devs.items():
            print("{:<30} {:>10.2f}".format(display_mapping.get(col, col), sigma))


        # Brug 1.5 * sigma som en typisk "god" tolerance for advarsel om stram pasform
        chest_sigma = std_devs.get('ChestWidth', 0.0)
        if chest_sigma > 0:
            suggested_tolerance = chest_sigma * 1.5
            print(
                f"For Brystmål (Chest), brug f.eks. 1.5 x σ = {suggested_tolerance:.2f} cm som din 'EXCESSIVE_FIT_TOLERANCE'.")


    except FileNotFoundError:
        print(f"Fejl: Filen '{file_path}' blev ikke fundet. Tjek stien.")
    except Exception as e:
        print(f"Der opstod en uventet fejl: {e}")


if __name__ == "__main__":
    calculate_std_devs(FILE_PATH)