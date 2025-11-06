import joblib
import pandas as pd
import numpy as np

# Load trained imputer
imputer = joblib.load("final_imputer.pkl")

input_df = # et eller andet fra Unity converted til en pandas Dataframe for compatibility.

# Drop gender fordi det ikke skal bruges af modellen, align columns så features er i den rigtige rækkefølge
numeric_cols_no_gender = [col for col in imputer.feature_names_in_]
input_df_aligned = input_df[numeric_cols_no_gender]

# Impute missing values
new_data_imputed = imputer.transform(input_df_aligned)

print(new_data_imputed)

