import os
import pandas as pd

def clean_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    data = pd.read_csv(file_path, encoding="latin1")  # Correct encoding
    print("Columns in CSV:", data.columns)  # Debugging: Print columns
    
    # Ensure the column exists before converting it to datetime
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'], format="%d/%m/%Y", dayfirst=True)
    else:
        raise KeyError("Column 'Date' not found in the CSV file")
    
    # Ensure the column exists before accessing it
    if 'AC Penrose PM10 24h average [µg/m³]' in data.columns:
        pm10_values = data[['Date', 'AC Penrose PM10 24h average [µg/m³]']].copy()
        data.drop("AC Penrose PM10 24h average [µg/m³]", axis=1, inplace=True)
    else:
        raise KeyError("Column 'AC Penrose PM10 24h average [µg/m³]' not found in the CSV file")
    
    data.set_index('Date', inplace=True)
    return data, pm10_values

# Example usage
# file_path = 'path_to_your_file.csv'
# data, pm10_values = clean_data(file_path)
