import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from .data_cleaner import clean_data

class PM10Predictor:
    def __init__(self, xgb_model_path, svm_model_path, rf_model_path, lstm_model_path, gru_model_path):
        self.custom_objects = {'mse': tf.keras.losses.MeanSquaredError()}  # Correct custom objects

        # Load models saved with pickle
        with open(xgb_model_path, 'rb') as file:
            self.xgb_model = pickle.load(file)

        with open(svm_model_path, 'rb') as file:
            self.svm_model = pickle.load(file)

        with open(rf_model_path, 'rb') as file:
            self.rf_model = pickle.load(file)

        # Load models saved in Keras format with custom objects
        self.lstm_model = load_model(lstm_model_path, custom_objects=self.custom_objects)
        self.gru_model = load_model(gru_model_path, custom_objects=self.custom_objects)

    def select_features(self, data, target_column, num_features):
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Train a Random Forest Regressor to determine feature importances
        rf = RandomForestRegressor()
        rf.fit(X, y)

        # Get the feature importances
        importances = rf.feature_importances_
        feature_indices = np.argsort(importances)[-num_features:]

        # Get the column names for the selected features
        selected_columns = X.columns[feature_indices]
        return selected_columns

    def predict_pm10_24hr_avg(self, data_path, num_features=5):
        # Load and clean data
        data, pm10_values = clean_data(data_path)

        # Print columns to debug the KeyError issue
        print("Columns in CSV after cleaning:", data.columns)

        # Identify the correct target column
        target_column = 'AC Penrose PM10 24h average [µg/m³]'
        if target_column not in data.columns:
            # Try finding the column by pattern matching or other heuristic
            possible_columns = [col for col in data.columns if 'PM10' in col]
            if possible_columns:
                target_column = possible_columns[0]
                print(f"Target column identified as: {target_column}")
            else:
                raise KeyError(f"Target column not found. Available columns: {data.columns}")

        # Keep a copy of the target column for later visualization
        pm10_copy = pm10_values.copy()

        # Determine the features to use using feature selection
        selected_features = self.select_features(data, target_column, num_features)

        # Select only these features for prediction
        features = data[selected_features]

        # Assuming your features are based on the last day's data
        last_date = data.index[-1]
        features = features.loc[last_date].values.reshape(1, -1)

        xgb_prediction = self.xgb_model.predict(features)
        svm_prediction = self.svm_model.predict(features)
        rf_prediction = self.rf_model.predict(features)

        # Assuming your LSTM/GRU models expect input shape (samples, time_steps, features)
        sequence_length = 10  # Example sequence length
        lstm_features = data[selected_features][-sequence_length:].values.reshape(1, sequence_length, -1)
        gru_features = data[selected_features][-sequence_length:].values.reshape(1, sequence_length, -1)

        lstm_prediction = self.lstm_model.predict(lstm_features)
        gru_prediction = self.gru_model.predict(gru_features)

        predictions = {
            "date": last_date + timedelta(days=1),
            "XGBoost": xgb_prediction[0],
            "SVM": svm_prediction[0],
            "Random Forest": rf_prediction[0],
            "LSTM": lstm_prediction[0][0],
            "GRU": gru_prediction[0][0]
        }

        return predictions, pm10_copy

    def plot_pm10_with_predictions(self, pm10_values, predictions):
        # Plot the last month's data
        pm10_values.set_index('Date', inplace=True)
        last_month_data = pm10_values[-30:]
        last_month_dates = last_month_data.index
        last_month_values = last_month_data.iloc[:, 0].values  # Get the first column as the PM10 values

        # Append the prediction date and value
        prediction_date = predictions['date']
        xgb_pred_value = predictions['XGBoost']
        rf_pred_value = predictions['Random Forest']
        gru_pred_value = predictions['GRU']

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(last_month_dates, last_month_values, label='Actual PM10 Values', marker='o')
        plt.axvline(prediction_date, color='grey', linestyle='--', label='Prediction Date')
        plt.plot([last_month_dates[-1], prediction_date], [last_month_values[-1], xgb_pred_value], label='XGBoost Prediction', marker='x')
        plt.plot([last_month_dates[-1], prediction_date], [last_month_values[-1], rf_pred_value], label='Random Forest Prediction', marker='s')
        plt.plot([last_month_dates[-1], prediction_date], [last_month_values[-1], gru_pred_value], label='GRU Prediction', marker='d')

        plt.xlabel('Date')
        plt.ylabel('PM10 Values')
        plt.title('Last Month PM10 Values and Next Day Predictions')
        plt.legend()
        plt.grid(True)
        plt.show()

# Usage example
data_path = 'data/PM10_24HR_input.csv'
xgb_model_path = 'models/best_xgb_model2.pkl'
svm_model_path = 'models/best_svm_model.pkl'
rf_model_path = 'models/best_rf_model.pkl'
lstm_model_path = 'models/lstm_model.keras'
gru_model_path = 'models/gru_model.h5'

predictor = PM10Predictor(xgb_model_path, svm_model_path, rf_model_path, lstm_model_path, gru_model_path)
predictions, pm10_values = predictor.predict_pm10_24hr_avg(data_path)
print(f"Predicted PM10 24hr avg value for {predictions['date']}:")
print(f"XGBoost: {predictions['XGBoost']}")
print(f"SVM: {predictions['SVM']}")
print(f"Random Forest: {predictions['Random Forest']}")
print(f"GRU: {predictions['GRU']}")

predictor.plot_pm10_with_predictions(pm10_values, predictions)
56
+9-