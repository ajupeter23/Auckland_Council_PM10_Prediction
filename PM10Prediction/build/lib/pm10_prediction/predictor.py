import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf
from .data_cleaner import clean_data

class PM10Predictor:
    def __init__(self, xgb_model_path, svm_model_path, rf_model_path, lstm_model_path, gru_model_path):
        self.custom_objects = {'mse': 'mean_squared_error'}

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

    def predict_pm10_24hr_avg(self, data_path):
        # Load and clean data
        data, pm10_values = clean_data(data_path)

        # Assuming your features are based on the last day's data
        last_date = data.index[-1]
        features = data.loc[last_date].values.reshape(1, -1)

        xgb_prediction = self.xgb_model.predict(features)
        svm_prediction = self.svm_model.predict(features)
        rf_prediction = self.rf_model.predict(features)

        # Assuming your LSTM/GRU models expect input shape (samples, time_steps, features)
        sequence_length = 10  # Example sequence length
        lstm_features = data[-sequence_length:].values.reshape(1, sequence_length, -1)
        gru_features = data[-sequence_length:].values.reshape(1, sequence_length, -1)

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

        return predictions, pm10_values

    def plot_pm10_with_predictions(self, pm10_values, predictions):
        # Plot the last month's data
        pm10_values.set_index('Date', inplace=True)
        last_month_data = pm10_values[-30:]
        last_month_dates = last_month_data.index
        last_month_values = last_month_data['Penrose PM10 24h average [µg/m³]'].values

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
