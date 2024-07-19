import os
import pandas as pd

def clean_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    data = pd.read_csv(file_path, encoding="latin")
    print("Columns in CSV:", data.columns)  # Debugging: Print columns
    
    # Specify the correct date format
    data['Date'] = pd.to_datetime(data['Date'], format="%d/%m/%Y", dayfirst=True)

    # Ensure the column exists before accessing it
    if 'Penrose PM10 24h average [µg/m³]' in data.columns:
        pm10_values = data[['Date', 'Penrose PM10 24h average [µg/m³]']].copy()
        data.drop("Penrose PM10 24h average [µg/m³]", axis=1, inplace=True)
    else:
        raise KeyError("Column 'Penrose PM10 24h average [µg/m³]' not found in the CSV file")
    
    data.set_index('Date', inplace=True)
    return data, pm10_values
