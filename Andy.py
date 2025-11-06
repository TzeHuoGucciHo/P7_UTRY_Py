import joblib
import pandas as pd

# Load trained model/imputer
imputer = joblib.load("final_imputer.pkl")

# Impute missing values in new data
new_data_imputed = imputer.transform(input_data[num_cols_no_gender])
