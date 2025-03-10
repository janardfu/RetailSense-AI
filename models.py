import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from models.advanced_models import InventoryAIEngine

def train_prophet_model(data, forecast_period):
    """Train and generate forecast using Facebook Prophet"""
    # Prepare data for Prophet
    df = data.rename(columns={'date': 'ds', 'sales': 'y'})
    
    # Initialize and train model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True
    )
    model.fit(df)
    
    # Make forecast
    future = model.make_future_dataframe(periods=forecast_period)
    forecast = model.predict(future)
    
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def train_arima_model(data, forecast_period):
    """Train and generate forecast using ARIMA"""
    # Prepare data
    series = data['sales']
    
    # Fit ARIMA model
    model = ARIMA(series, order=(1, 1, 1))
    model_fit = model.fit()
    
    # Generate forecast
    forecast = model_fit.forecast(steps=forecast_period)
    
    # Create forecast dataframe
    forecast_dates = pd.date_range(
        start=data['date'].max(),
        periods=forecast_period + 1,
        freq='D'
    )[1:]
    
    forecast_df = pd.DataFrame({
        'ds': forecast_dates,
        'yhat': forecast
    })
    
    return forecast_df

def calculate_forecast_accuracy(forecast, actual):
    """Calculate forecast accuracy metrics"""
    y_true = actual['sales'].values
    y_pred = forecast['ensemble'][:len(y_true)]
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Calculate MAPE
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE (%)': mape
    }

def train_lstm_model(data, forecast_period):
    """Train and generate forecast using LSTM"""
    model = LSTMForecastModel(sequence_length=30)
    model.fit(data['sales'].values)
    forecast = model.predict(data['sales'].values, forecast_period)
    
    # Create forecast dataframe
    forecast_dates = pd.date_range(
        start=data['date'].max(),
        periods=forecast_period + 1,
        freq='D'
    )[1:]
    
    forecast_df = pd.DataFrame({
        'ds': forecast_dates,
        'yhat': forecast
    })
    
    return forecast_df

def train_ensemble_model(data, forecast_period):
    """Train and generate forecast using Ensemble"""
    model = EnsembleForecastModel()
    model.fit(data['sales'].values)
    forecast = model.predict(data['sales'].values, forecast_period)
    
    # Create forecast dataframe
    forecast_dates = pd.date_range(
        start=data['date'].max(),
        periods=forecast_period + 1,
        freq='D'
    )[1:]
    
    forecast_df = pd.DataFrame({
        'ds': forecast_dates,
        'yhat': forecast
    })
    
    return forecast_df 