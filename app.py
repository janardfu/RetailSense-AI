import os
import warnings
import tensorflow as tf
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from database import MongoDB
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import io
import joblib
import logging
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
from models import InventoryManager
import random
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define currency symbols
CURRENCY_SYMBOLS = {
    'INR': '₹',
    'USD': '$',
    'EUR': '€',
    'GBP': '£'
}

# Define sample products for data generation
SAMPLE_PRODUCTS = {
    "Electronics": [
        {"name": "Smart LED TV", "price": 45999.99},
        {"name": "Laptop", "price": 65999.99},
        {"name": "Smartphone", "price": 24999.99},
        {"name": "Tablet", "price": 29999.99},
        {"name": "Wireless Earbuds", "price": 4999.99},
        {"name": "Smart Watch", "price": 12999.99},
        {"name": "Gaming Console", "price": 39999.99},
        {"name": "Digital Camera", "price": 34999.99},
        {"name": "Bluetooth Speaker", "price": 3999.99},
        {"name": "Power Bank", "price": 1999.99}
    ],
    "Mobile Accessories": [
        {"name": "Phone Case", "price": 999.99},
        {"name": "Screen Protector", "price": 499.99},
        {"name": "Car Charger", "price": 799.99},
        {"name": "USB Cable", "price": 299.99},
        {"name": "Phone Stand", "price": 599.99},
        {"name": "Wireless Charger", "price": 1499.99},
        {"name": "Selfie Stick", "price": 699.99},
        {"name": "Memory Card", "price": 899.99},
        {"name": "Phone Grip", "price": 399.99},
        {"name": "Charging Adapter", "price": 999.99}
    ],
    "Home Appliances": [
        {"name": "Microwave Oven", "price": 8999.99},
        {"name": "Air Conditioner", "price": 35999.99},
        {"name": "Refrigerator", "price": 25999.99},
        {"name": "Washing Machine", "price": 22999.99},
        {"name": "Water Purifier", "price": 15999.99},
        {"name": "Air Purifier", "price": 12999.99},
        {"name": "Vacuum Cleaner", "price": 9999.99},
        {"name": "Dishwasher", "price": 28999.99},
        {"name": "Electric Kettle", "price": 1499.99},
        {"name": "Induction Cooktop", "price": 3999.99}
    ],
    "Computer Parts": [
        {"name": "SSD Drive", "price": 4999.99},
        {"name": "RAM Module", "price": 3499.99},
        {"name": "Graphics Card", "price": 29999.99},
        {"name": "Processor", "price": 24999.99},
        {"name": "Motherboard", "price": 12999.99},
        {"name": "Power Supply", "price": 4999.99},
        {"name": "Computer Case", "price": 3999.99},
        {"name": "CPU Cooler", "price": 2999.99},
        {"name": "Hard Drive", "price": 3999.99},
        {"name": "Network Card", "price": 1499.99}
    ],
    "Audio Devices": [
        {"name": "Headphones", "price": 7999.99},
        {"name": "Sound Bar", "price": 14999.99},
        {"name": "Home Theater", "price": 29999.99},
        {"name": "Microphone", "price": 4999.99},
        {"name": "DJ Controller", "price": 19999.99},
        {"name": "Audio Interface", "price": 12999.99},
        {"name": "Studio Monitors", "price": 24999.99},
        {"name": "MIDI Keyboard", "price": 8999.99},
        {"name": "Amplifier", "price": 9999.99},
        {"name": "Portable Speaker", "price": 5999.99}
    ]
}

# Filter TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def prepare_lstm_data(data, lookback=30):
    """
    Prepare data for LSTM model training
    
    Args:
        data (DataFrame): Input data containing stock levels and other features
        lookback (int): Number of previous time steps to use for prediction
        
    Returns:
        X (array): Input sequences
        y (array): Target values
        scaler (StandardScaler): Fitted scaler for inverse transformation
    """
    try:
        # Ensure data has datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'date' in data.columns:
                data = data.set_index('date')
            elif 'timestamp' in data.columns:
                data = data.set_index('timestamp')
            else:
                st.error("No datetime column found in the data")
                return None, None, None
        
        # Sort index to ensure chronological order
        data = data.sort_index()
        
        # Use only available features
        base_features = ['stock_level', 'price', 'device_usage_hours']
        available_features = [f for f in base_features if f in data.columns]

        if not available_features:
            st.error("No features available for LSTM model")
            return None, None, None

        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data[available_features])
        
        return scaled_data, available_features, scaler
        
    except Exception as e:
        st.error(f"Error preparing LSTM data: {str(e)}")
        return None, None, None

def analyze_demand_patterns(historical_data):
    if historical_data is None or historical_data.empty or 'stock_level' not in historical_data.columns:
        st.warning("Insufficient data for demand pattern analysis")
        return
        
    with st.spinner('Analyzing demand patterns...'):
        try:
            # Ensure proper datetime index
            if not isinstance(historical_data.index, pd.DatetimeIndex):
                if 'date' in historical_data.columns:
                    historical_data = historical_data.set_index('date')
                elif 'timestamp' in historical_data.columns:
                    historical_data = historical_data.set_index('timestamp')
                else:
                    st.error("No datetime column found in the data")
                    return
            
            # Sort index and resample data to daily frequency
            historical_data = historical_data.sort_index()
            ts_data = historical_data['stock_level'].resample('D').mean()
            
            # Handle missing values
            ts_data = ts_data.fillna(method='ffill').fillna(method='bfill')
            
            if len(ts_data) < 14:
                st.warning("Insufficient data for pattern analysis. Need at least 14 days of data.")
                return
                
            try:
                # Decomposition Analysis with error handling
                period = min(7, len(ts_data) // 2)  # Adjust period based on data length
                decomposition = seasonal_decompose(ts_data, period=period)
                
                # Create subplots for components
                fig = go.Figure()
                
                # Original Data
                fig.add_trace(go.Scatter(
                    x=ts_data.index,
                    y=ts_data.values,
                    name='Original',
                    line=dict(color='blue')
                ))
                
                # Trend
                fig.add_trace(go.Scatter(
                    x=ts_data.index,
                    y=decomposition.trend.fillna(method='ffill').fillna(method='bfill'),
                    name='Trend',
                    line=dict(color='red')
                ))
                
                # Seasonal
                fig.add_trace(go.Scatter(
                    x=ts_data.index,
                    y=decomposition.seasonal.fillna(method='ffill').fillna(method='bfill'),
                    name='Seasonal',
                    line=dict(color='green')
                ))
                
                # Residual
                fig.add_trace(go.Scatter(
                    x=ts_data.index,
                    y=decomposition.resid.fillna(method='ffill').fillna(method='bfill'),
                    name='Residual',
                    line=dict(color='gray')
                ))
                
                fig.update_layout(
                    title='Time Series Decomposition',
                    xaxis_title='Date',
                    yaxis_title='Stock Level',
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate trend direction and strength with error handling
                trend_values = decomposition.trend.fillna(method='ffill').fillna(method='bfill')
                if len(trend_values) >= 7:
                    trend_direction = "Increasing" if trend_values.iloc[-1] > trend_values.iloc[-7] else "Decreasing"
                    trend_strength = abs(trend_values.iloc[-1] - trend_values.iloc[-7]) / trend_values.iloc[-7] * 100
                else:
                    trend_direction = "Insufficient data"
                    trend_strength = 0
                
                # Calculate seasonality strength with error handling
                seasonal_values = decomposition.seasonal.fillna(method='ffill').fillna(method='bfill')
                if len(seasonal_values) > 0 and len(ts_data) > 0:
                    seasonality_strength = np.std(seasonal_values) / np.std(ts_data) * 100
                else:
                    seasonality_strength = 0
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Trend Direction",
                        trend_direction,
                        f"{trend_strength:.1f}% change over 7 days" if trend_direction != "Insufficient data" else None
                    )
                    
                with col2:
                    st.metric(
                        "Seasonality Strength",
                        f"{seasonality_strength:.1f}%",
                        help="Higher values indicate stronger seasonal patterns"
                    )
                
                # Weekly Pattern Analysis
                weekly_pattern = ts_data.groupby(ts_data.index.dayofweek).mean()
                
                fig_weekly = go.Figure(go.Bar(
                    x=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                    y=weekly_pattern.values,
                    text=np.round(weekly_pattern.values, 1),
                    textposition='auto',
                ))
                
                fig_weekly.update_layout(
                    title='Average Stock Level by Day of Week',
                    xaxis_title='Day of Week',
                    yaxis_title='Average Stock Level',
                    height=400
                )
                
                st.plotly_chart(fig_weekly, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error in decomposition analysis: {str(e)}")
                st.info("Unable to perform detailed trend analysis. This might be due to insufficient or irregular data.")
                
                # Fallback to simple trend analysis
                if len(ts_data) >= 7:
                    simple_trend = ts_data.rolling(window=7).mean()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=ts_data.index,
                        y=ts_data.values,
                        name='Original',
                        line=dict(color='blue')
                    ))
                    fig.add_trace(go.Scatter(
                        x=simple_trend.index,
                        y=simple_trend.values,
                        name='7-day Moving Average',
                        line=dict(color='red')
                    ))
                    
                    fig.update_layout(
                        title='Simple Trend Analysis (7-day Moving Average)',
                        xaxis_title='Date',
                        yaxis_title='Stock Level',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error in demand pattern analysis: {str(e)}")
            st.info("Please ensure your data contains proper datetime information and sufficient historical records")

def generate_sample_data(db, num_products, min_daily_sales, max_daily_sales, days_of_history, restock_probability):
    """
    Generate sample data for the inventory system
    
    Args:
        db: MongoDB instance
        num_products: Number of products per category
        min_daily_sales: Minimum number of sales per day
        max_daily_sales: Maximum number of sales per day
        days_of_history: Number of days of historical data to generate
        restock_probability: Probability of restocking per day
    """
    try:
        # Ensure minimum days for LSTM
        if days_of_history < 60:
            days_of_history = 60
            logger.warning("Adjusted days_of_history to minimum 60 days required for LSTM")
        
        # Clear existing data
        db.clear_all_data()
        
        # Generate products from each category
        for category, products in SAMPLE_PRODUCTS.items():
            selected_products = random.sample(products, min(num_products, len(products)))
            for product in selected_products:
                initial_stock = random.randint(100, 300)  # Higher initial stock
                product_data = {
                    "product_name": product['name'],
                    "name": product['name'],
                    "category": category,
                    "price": product['price'],
                    "stock_level": initial_stock,
                    "quantity": initial_stock,
                    "min_stock": 20,
                    "max_stock": 400,
                    "date": datetime.utcnow()
                }
                db.save_product(product_data)
        
        # Generate historical data
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_of_history)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Get all products
        products = db.get_all_products()
        if not products:
            raise ValueError("No products available")
        
        # Generate daily data with multiple sales per day to ensure enough data points
        for current_date in date_range:
            # Generate 2-5 sales events per day to ensure enough data points
            daily_sales = random.randint(max(2, min_daily_sales), max(5, max_daily_sales))
            
            for hour in range(9, 21, 24//daily_sales):  # Spread sales throughout the day
                # Generate sale timestamp
                sale_time = current_date + timedelta(
                    hours=hour,
                    minutes=random.randint(0, 59)
                )
                
                # Create sale data
                sale_data = {
                    'customer': {
                        'name': f'Customer_{sale_time.strftime("%Y%m%d_%H%M%S")}',
                        'email': f'customer_{sale_time.strftime("%Y%m%d")}@example.com',
                        'phone': f'{random.randint(1000000000, 9999999999)}'
                    },
                    'items': [],
                    'total_amount': 0,
                    'payment_method': random.choice(['Cash', 'Card', 'UPI', 'Net Banking']),
                    'timestamp': sale_time,
                    'created_at': sale_time,
                    'updated_at': sale_time,
                    'status': 'completed'
                }
                
                # Add items to sale
                num_items = random.randint(1, 3)
                sale_products = random.sample(products, min(num_items, len(products)))
                
                for product in sale_products:
                    quantity = random.randint(1, 3)
                    price = float(product['price'])
                    amount = quantity * price
                    
                    sale_data['items'].append({
                        'product_id': str(product['_id']),
                        'name': product.get('name', product.get('product_name')),
                        'quantity': quantity,
                        'price': price,
                        'amount': amount
                    })
                    sale_data['total_amount'] += amount
                
                sale_data['total_amount'] = round(sale_data['total_amount'], 2)
                
                try:
                    # Save sale and update inventory
                    sale_id = db.save_sales_data(sale_data)
                    
                    # Update product quantities
                    for item in sale_data['items']:
                        db.update_product(
                            item['product_id'],
                            {
                                'quantity_change': -item['quantity'],
                                'updated_at': sale_time
                            }
                        )
                except Exception as e:
                    logger.error(f"Error saving sale data: {str(e)}")
                    continue
            
            # Daily restock check
            if random.random() < restock_probability:
                for product in products:
                    current_stock = product.get('stock_level', product.get('quantity', 0))
                    if current_stock < product.get('min_stock', 50):
                        restock_qty = random.randint(50, 200)
                        db.update_product(
                            str(product['_id']),
                            {
                                'quantity_change': restock_qty,
                                'updated_at': current_date
                            }
                        )
        
        logger.info(f"Generated {days_of_history} days of historical data with multiple daily sales")
        return True, "Sample data generated successfully"
        
    except Exception as e:
        logger.error(f"Error generating sample data: {str(e)}")
        return False, str(e)

def create_lstm_model(lookback, n_features, layers=2, units=50):
    """
    Create an LSTM model with specified architecture
    
    Args:
        lookback: Number of previous time steps to use
        n_features: Number of input features
        layers: Number of LSTM layers
        units: Number of units per layer
    
    Returns:
        model: Compiled LSTM model
    """
    model = Sequential()
    
    # First LSTM layer
    model.add(LSTM(units=units, return_sequences=(layers > 1), input_shape=(lookback, n_features)))
    model.add(Dropout(0.2))
    
    # Additional LSTM layers
    for i in range(layers - 2):
        model.add(LSTM(units=units, return_sequences=True))
        model.add(Dropout(0.2))
    
    # Last LSTM layer
    if layers > 1:
        model.add(LSTM(units=units))
        model.add(Dropout(0.2))
    
    # Output layer
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mse')
    return model

def load_or_train_lstm_model(historical_data, product_id, lookback=30, layers=2, units=50, force_retrain=False):
    """
    Load existing LSTM model or train a new one if needed
    
    Args:
        historical_data: Array-like data with historical stock data
        product_id: ID of the product to model
        lookback: Number of previous time steps to use
        layers: Number of LSTM layers
        units: Number of units per layer
        force_retrain: Whether to force retraining even if model exists
    
    Returns:
        model: Trained LSTM model
        scaler: Fitted StandardScaler
    """
    try:
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        model_path = f'models/lstm_{product_id}.h5'
        scaler_path = f'models/scaler_{product_id}.pkl'
        
        # Check if model exists and load if not force_retrain
        if os.path.exists(model_path) and os.path.exists(scaler_path) and not force_retrain:
            logger.info(f"Loading existing model for product {product_id}")
            model = load_model(model_path)
            scaler = joblib.load(scaler_path)
            return model, scaler
        
        # Prepare data for training
        if historical_data is None or len(historical_data) == 0:
            raise ValueError("No historical data available for training")
        
        # Scale the data
        scaler = StandardScaler()
        # Ensure data is 2D for StandardScaler
        if len(historical_data.shape) == 1:
            scaled_data = scaler.fit_transform(historical_data.reshape(-1, 1))
        else:
            scaled_data = scaler.fit_transform(historical_data)
        
        # Get number of features
        n_features = scaled_data.shape[1] if len(scaled_data.shape) > 1 else 1
        
        # Create sequences for LSTM
        X, y = [], []
        for i in range(len(scaled_data) - lookback):
            # For multiple features, take all columns for X
            X.append(scaled_data[i:(i + lookback)])
            # For y, take only the first feature (stock level) as target
            y.append(scaled_data[i + lookback, 0])
        
        X = np.array(X)
        y = np.array(y)
        
        if len(X) < lookback:
            raise ValueError(f"Insufficient data for training. Need at least {lookback*2} points, got {len(historical_data)}")
        
        # Create and train model with correct number of features
        model = create_lstm_model(lookback, n_features, layers, units)
        
        # Train with early stopping
        model.fit(
            X, y,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
            ],
            verbose=0
        )
        
        # Save model and scaler
        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        
        logger.info(f"Successfully trained and saved model for product {product_id}")
        return model, scaler
        
    except Exception as e:
        logger.error(f"Error in load_or_train_lstm_model: {str(e)}")
        raise Exception(f"Failed to load or train LSTM model: {str(e)}")

# Page configuration
st.set_page_config(
    page_title="RetailSense Dashboard",
    page_icon="📊",
    layout="wide"
)

# Initialize MongoDB connection
db = MongoDB()
load_dotenv()

# Suppress TensorFlow warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Add after the page configuration
if 'currency' not in st.session_state:
    st.session_state.currency = 'INR'
    st.session_state.currency_symbol = '₹'

# Dashboard Layout Configuration
def render_dashboard():
    st.title("RetailSense Dashboard")

    # Get current data
    inventory_data = db.get_current_stock_levels()
    if inventory_data is None or inventory_data.empty:
        st.warning("No inventory data available. Please add data through the Inventory Management page.")
        return

    # Add category if missing
    if 'category' not in inventory_data.columns:
        inventory_data['category'] = 'Default'

    # Layout: Top Level KPIs
    st.subheader("📈 Key Performance Indicators")
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        total_products = len(inventory_data)
        total_categories = len(inventory_data['category'].unique())
        st.metric(
            label="Total Products",
            value=total_products,
            delta=f"{total_categories} Categories",
            help="Total number of unique products and categories in inventory"
        )

    with kpi_col2:
        if 'stock_level' in inventory_data.columns:
            total_stock = inventory_data['stock_level'].sum()
            avg_stock = inventory_data['stock_level'].mean()
            st.metric(
                label="Total Stock",
                value=f"{total_stock:,.0f}",
                delta=f"Avg: {avg_stock:.1f} per product",
                help="Total units in stock across all products"
            )

    with kpi_col3:
        if 'price' in inventory_data.columns and 'stock_level' in inventory_data.columns:
            total_value = (inventory_data['price'] * inventory_data['stock_level']).sum()
            avg_value = total_value / total_products if total_products > 0 else 0
            st.metric(
                label="Inventory Value",
                value=f"{st.session_state.currency_symbol}{total_value:,.2f}",
                delta=f"Avg: {st.session_state.currency_symbol}{avg_value:,.2f} per product",
                help="Total value of current inventory"
            )

    with kpi_col4:
        if 'stock_level' in inventory_data.columns:
            low_stock_count = len(inventory_data[inventory_data['stock_level'] < 50])
            st.metric(
                label="Low Stock Items",
                value=low_stock_count,
                delta=f"{(low_stock_count/total_products*100):.1f}% of total" if total_products > 0 else "N/A",
                delta_color="inverse",
                help="Products with stock level below 50 units"
            )

    # Layout: Main Dashboard Sections
    st.write("")  # Add spacing
    
    # Row 1: Stock Analysis
    stock_col1, stock_col2 = st.columns(2)
    
    with stock_col1:
        # Stock Health Distribution
        if 'stock_level' in inventory_data.columns:
            stock_health = pd.cut(
                inventory_data['stock_level'],
                bins=[0, 10, 50, 100, float('inf')],
                labels=['Critical', 'Low', 'Healthy', 'Excess']
            )
            health_counts = stock_health.value_counts()
            
            fig_health = px.pie(
                values=health_counts.values,
                names=health_counts.index,
                title='Stock Health Distribution',
                color=health_counts.index,
                color_discrete_map={
                    'Critical': '#ff0000',
                    'Low': '#ffa500',
                    'Healthy': '#00ff00',
                    'Excess': '#0000ff'
                }
            )
            fig_health.update_layout(height=400)
            fig_health.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_health, use_container_width=True)
    
    with stock_col2:
        # Stock vs Price Analysis
        if 'price' in inventory_data.columns and 'stock_level' in inventory_data.columns:
            inventory_data['revenue_potential'] = inventory_data['price'] * inventory_data['stock_level']
            
            fig_bubble = px.scatter(
                inventory_data,
                x='price',
                y='stock_level',
                size='revenue_potential',
                color='category',
                hover_name='product_name',
                title='Stock vs Price Analysis',
                labels={
                    'price': f'Price ({st.session_state.currency_symbol})',
                    'stock_level': 'Current Stock',
                    'revenue_potential': 'Revenue Potential'
                }
            )
            fig_bubble.update_layout(height=400)
            st.plotly_chart(fig_bubble, use_container_width=True)

    # Row 2: Category Analysis
    st.subheader("📊 Category Analysis")
    cat_col1, cat_col2 = st.columns(2)
    
    with cat_col1:
        # Category Value Distribution
        if not inventory_data.empty and 'category' in inventory_data.columns:
            category_value = inventory_data.groupby('category').apply(
                lambda x: (x['price'] * x['stock_level']).sum()
            )
            fig_value = go.Figure(go.Bar(
                x=category_value.values,
                y=category_value.index,
                orientation='h',
                text=[f"{st.session_state.currency_symbol}{x:,.0f}" for x in category_value.values],
                textposition='auto',
            ))
            
            fig_value.update_layout(
                title='Inventory Value by Category',
                xaxis_title='Total Value',
                yaxis_title='Category',
                height=400
            )
            st.plotly_chart(fig_value, use_container_width=True)
    
    with cat_col2:
        # Product Count and Average Price by Category
        if not inventory_data.empty and 'category' in inventory_data.columns:
            category_stats = inventory_data.groupby('category').agg({
                'product_name': 'count',
                'price': 'mean'
            }).round(2)
            
            fig_stats = go.Figure()
            
            fig_stats.add_trace(go.Bar(
                name='Product Count',
                x=category_stats.index,
                y=category_stats['product_name'],
                yaxis='y',
                offsetgroup=1,
                text=category_stats['product_name'],
                textposition='auto',
            ))
            
            fig_stats.add_trace(go.Scatter(
                name='Average Price',
                x=category_stats.index,
                y=category_stats['price'],
                yaxis='y2',
                mode='lines+markers',
                line=dict(color='red'),
                text=[f"{st.session_state.currency_symbol}{x:,.0f}" for x in category_stats['price']],
            ))
            
            fig_stats.update_layout(
                title='Product Count and Average Price by Category',
                yaxis=dict(title='Number of Products', side='left'),
                yaxis2=dict(title='Average Price', side='right', overlaying='y'),
                height=400,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            st.plotly_chart(fig_stats, use_container_width=True)

    # Row 3: Sales and Performance
    st.subheader("📈 Sales and Performance")
    sales_col1, sales_col2 = st.columns(2)
    
    with sales_col1:
        # Daily Sales Trend
        try:
            sales_data = db.get_historical_stock_data()
            if not sales_data.empty and 'total_amount' in sales_data.columns:
                daily_sales = sales_data.groupby(sales_data.index.date)['total_amount'].sum().reset_index()
                fig_daily_sales = px.line(
                    daily_sales,
                    x='index',
                    y='total_amount',
                    title='Daily Sales Trend',
                    labels={'index': 'Date', 'total_amount': 'Sales Amount'}
                )
                fig_daily_sales.update_layout(height=400)
                st.plotly_chart(fig_daily_sales, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating sales trend chart: {str(e)}")
    
    with sales_col2:
        # Payment Method Distribution
        try:
            if not sales_data.empty and 'payment_method' in sales_data.columns:
                payment_dist = sales_data['payment_method'].value_counts()
                fig_payment = px.pie(
                    values=payment_dist.values,
                    names=payment_dist.index,
                    title='Payment Method Distribution'
                )
                fig_payment.update_layout(height=400)
                st.plotly_chart(fig_payment, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating payment distribution chart: {str(e)}")

    # Row 4: Insights and Recommendations
    st.subheader("💡 Insights and Recommendations")
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.info("""
        **Stock Health Summary:**
        - Critical items require immediate attention
        - Low stock items need reordering soon
        - Healthy items are at optimal levels
        - Excess items may need promotions
        """)
    
    with insight_col2:
        st.info("""
        **Optimization Opportunities:**
        - Balance stock levels across categories
        - Focus on high-turnover items
        - Review pricing strategy for slow-moving items
        - Consider seasonal trends in restocking
        """)

# Sidebar navigation
st.sidebar.title("AI Inventory Manager")
page = st.sidebar.selectbox(
    "Navigation",
    ["Dashboard",
    "Inventory Management",
    "AI Predictions",
    "Optimization",
     "Settings"]
)

# Main content
if page == "Dashboard":
    render_dashboard()

elif page == "AI Predictions":
    st.title("AI-Powered Predictions")

    # Check if there's data in the database first
    historical_data = db.get_historical_stock_data()

    if historical_data is None or historical_data.empty or (isinstance(historical_data, pd.DataFrame) and len(historical_data) < 60):
        st.warning("""
        Insufficient historical data for AI predictions. You need at least 60 days of data.
        Current data points: {}
        
        Please generate more sample data to use the AI predictions feature.
        """.format(len(historical_data) if historical_data is not None and not historical_data.empty else 0))
        
        # Add a button to generate required data
        if st.button("Generate Required Sample Data", type="primary"):
            try:
                with st.spinner("Generating comprehensive sample data for AI predictions..."):
                    # Generate sample data with optimal settings for LSTM
                    success, message = generate_sample_data(
                        db=db,
                        num_products=3,  # Reasonable number of products per category
                        min_daily_sales=3,  # Ensure multiple data points per day
                        max_daily_sales=5,
                        days_of_history=90,  # More than minimum required 60 days
                        restock_probability=0.2
                    )
                    
                    if success:
                        st.success("""
                        ✅ Sample data generated successfully!
                        - 90 days of historical data
                        - Multiple daily sales records
                        - Sufficient data points for AI predictions
                        
                        Please refresh the page to start using AI predictions.
                        """)
                        # Add a refresh button
                        if st.button("Refresh Page"):
                            st.rerun()
                    else:
                        st.error(f"Failed to generate sample data: {message}")
            except Exception as e:
                st.error(f"Error generating sample data: {str(e)}")
    else:
        # Rest of the existing AI Predictions code
        prediction_state = st.empty()
        with prediction_state.container():
            with st.spinner("Loading predictions... This may take a few moments."):
                # Time Series Prediction
                st.subheader("Stock Level Predictions")

                # Get historical data with caching
                @st.cache_data(ttl=3600)  # Cache for 1 hour
                def get_cached_historical_data():
                    data = db.get_historical_stock_data()
                    if data is None or data.empty:
                        return None
                    return data

                historical_data = get_cached_historical_data()

                if historical_data is not None and not historical_data.empty:
                    # Verify required columns exist
                    required_columns = ['stock_level', 'price']
                    if not all(col in historical_data.columns for col in required_columns):
                        st.error("Missing required columns in historical data. Please regenerate sample data.")
                        st.stop()

                    # Add a progress bar
                    progress_bar = st.progress(0)

                    try:
                        # Prepare data for LSTM with caching
                        @st.cache_data(ttl=3600)
                        def get_cached_lstm_data(data):
                            if data is None or data.empty:
                                return None, None, None
                            return prepare_lstm_data(data)

                        scaled_data, features, scaler = get_cached_lstm_data(historical_data)
                        progress_bar.progress(33)

                        if scaled_data is not None and len(scaled_data) > 0:
                            # Load or train model with caching
                            @st.cache_resource(ttl=3600)
                            def get_cached_lstm_model(data, product_id="default"):
                                if data is None or len(data) == 0:
                                    return None, None
                                try:
                                    return load_or_train_lstm_model(data, product_id)
                                except Exception as e:
                                    st.error(f"Error loading/training LSTM model: {str(e)}")
                                    return None, None

                            model, model_scaler = get_cached_lstm_model(scaled_data)
                            progress_bar.progress(66)

                            if model is not None:
                                # Make predictions
                                future_days = st.slider("Prediction Horizon (Days)", 7, 30, 14)

                                # Calculate predictions
                                @st.cache_data(ttl=3600)
                                def calculate_predictions(_model, _scaler, last_sequence, days, n_features):
                                    if last_sequence is None or len(last_sequence) == 0:
                                        return None
                                    try:
                                        predictions = []
                                        current_sequence = last_sequence.copy()

                                        for _ in range(days):
                                            # Reshape sequence for prediction
                                            pred_input = current_sequence.reshape(1, current_sequence.shape[0], n_features)
                                            pred = _model.predict(pred_input)
                                            predictions.append(pred[0, 0])
                                            
                                            # Update sequence
                                            current_sequence = np.roll(current_sequence, -1, axis=0)
                                            # Update only the stock level feature (first column)
                                            current_sequence[-1, 0] = pred[0, 0]

                                        # Create array for inverse transform
                                        pred_array = np.zeros((len(predictions), n_features))
                                        pred_array[:, 0] = predictions  # Set stock level predictions
                                        
                                        # Inverse transform
                                        return _scaler.inverse_transform(pred_array)[:, 0]
                                    except Exception as e:
                                        st.error(f"Error calculating predictions: {str(e)}")
                                        return None

                                n_features = scaled_data.shape[1]
                                lookback = 30  # Default lookback period
                                last_sequence = scaled_data[-lookback:]
                                
                                predictions = calculate_predictions(
                                    model, 
                                    scaler, 
                                    last_sequence, 
                                    future_days,
                                    n_features
                                )
                                progress_bar.progress(100)

                                if predictions is not None:
                                    # Plot predictions
                                    dates = pd.date_range(
                                        start=datetime.now(), periods=future_days)
                                    pred_df = pd.DataFrame({
                                        'Date': dates,
                                        'Predicted_Stock': predictions
                                    })

                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(
                                        x=historical_data.index[-30:],
                                        y=historical_data['stock_level'][-30:],
                                        name='Historical'
                                    ))
                                    fig.add_trace(go.Scatter(
                                        x=pred_df['Date'],
                                        y=pred_df['Predicted_Stock'],
                                        name='Predicted',
                                        line=dict(dash='dash')
                                    ))

                                    fig.update_layout(
                                        title='Stock Level Forecast',
                                        xaxis_title='Date',
                                        yaxis_title='Stock Level'
                                    )
                                    st.plotly_chart(
                                        fig, use_container_width=True)

                                    # Add trend analysis
                                    st.subheader("📈 Trend Analysis")
                                    col1, col2 = st.columns(2)

                                    with col1:
                                        # Calculate trend metrics
                                        current_stock = historical_data['stock_level'][-1]
                                        avg_prediction = np.mean(predictions)
                                        trend_percentage = (
                                            (avg_prediction - current_stock) / current_stock) * 100

                                        st.metric(
                                            label="Trend Direction",
                                            value="Increasing" if trend_percentage > 0 else "Decreasing",
                                            delta=f"{trend_percentage:.1f}%"
                                        )

                                        # Volatility analysis
                                        volatility = np.std(
                                            predictions) / np.mean(predictions) * 100
                                        st.metric(
                                            label="Prediction Volatility",
                                            value=f"{volatility:.1f}%",
                                            help="Lower values indicate more stable predictions"
                                        )

                                    with col2:
                                        # Calculate min and max predictions
                                        min_pred = np.min(predictions)
                                        max_pred = np.max(predictions)
                                        range_percentage = (
                                            (max_pred - min_pred) / min_pred) * 100

                                        st.metric(
                                            label="Prediction Range",
                                            value=f"{range_percentage:.1f}%",
                                            help="Range between minimum and maximum predictions"
                                        )

                                        # Confidence score based on volatility
                                        confidence_score = max(
                                            0, min(100, 100 - volatility))
                                        # Remove the extra closing parenthesis
                                        st.metric(
                                            label="Prediction Confidence",
                                            value=f"{confidence_score:.1f}%",
                                            help="Higher values indicate more reliable predictions"
                                        )

                                    # Add improvement recommendations
                                    st.subheader(
                                        "🎯 Improvement Recommendations")

                                    recommendations = []

                                    # Analyze trend for recommendations
                                    if trend_percentage < -10:
                                        recommendations.append({
                                            "title": "📉 Stock Level Alert",
                                            "description": "Predicted significant decrease in stock levels. Consider increasing inventory or reviewing pricing strategy.",
                                            "priority": "High"
                                        })
                                    elif trend_percentage > 10:
                                        recommendations.append({
                                            "title": "📈 Overstocking Risk",
                                            "description": "Predicted significant increase in stock levels. Consider adjusting purchase orders or implementing promotions.",
                                            "priority": "Medium"
                                        })

                                    # Analyze volatility for recommendations
                                    if volatility > 20:
                                        recommendations.append({
                                            "title": "🎢 High Volatility Detected",
                                            "description": "Predictions show high variability. Consider implementing better forecasting methods or reviewing seasonal patterns.",
                                            "priority": "High"
                                        })

                                    # Add general recommendations
                                    recommendations.append({
                                        "title": "📊 Data Quality",
                                        "description": "Regular data updates and cleaning can improve prediction accuracy.",
                                        "priority": "Medium"
                                    })

                                    recommendations.append({
                                        "title": "🔄 Model Retraining",
                                        "description": "Consider retraining the model periodically to maintain prediction accuracy.",
                                        "priority": "Low"
                                    })

                                    # Display recommendations
                                    for rec in recommendations:
                                        with st.expander(f"{rec['title']} ({rec['priority']} Priority)"):
                                            st.write(rec['description'])

                                    # Add statistical insights
                                    st.subheader("📊 Statistical Insights")

                                    # Calculate statistical metrics
                                    stats_col1, stats_col2 = st.columns(2)

                                    with stats_col1:
                                        # Historical data statistics
                                        hist_mean = historical_data['stock_level'][-30:].mean(
                                        )
                                        hist_std = historical_data['stock_level'][-30:].std()
                                        st.write(
                                            "Historical Data (Last 30 Days)")
                                        st.metric(
                                            "Average Stock Level", f"{hist_mean:.1f}")
                                        st.metric(
                                            "Standard Deviation", f"{hist_std:.1f}")

                                    with stats_col2:
                                        # Prediction statistics
                                        pred_mean = np.mean(predictions)
                                        pred_std = np.std(predictions)
                                        st.write("Predictions")
                                        st.metric(
                                            "Average Predicted Level", f"{pred_mean:.1f}")
                                        st.metric(
                                            "Prediction Std Dev", f"{pred_std:.1f}")

                                    # Add seasonality detection
                                    st.subheader("🗓️ Seasonality Detection")

                                    # Simple seasonality detection based on
                                    # historical data
                                    # Last 90 days
                                    historical_series = historical_data['stock_level'][-90:]
                                    weekly_pattern = historical_series.groupby(
                                        historical_series.index.dayofweek).mean()
                                    monthly_pattern = historical_series.groupby(
                                        historical_series.index.month).mean()

                                    season_col1, season_col2 = st.columns(2)

                                    with season_col1:
                                        # Weekly patterns
                                        fig_weekly = go.Figure()
                                        fig_weekly.add_trace(go.Bar(
                                            x=['Mon', 'Tue', 'Wed', 'Thu',
                                                'Fri', 'Sat', 'Sun'],
                                            y=weekly_pattern.values,
                                            name='Average Stock Level'
                                        ))
                                        fig_weekly.update_layout(
                                            title='Weekly Pattern',
                                            xaxis_title='Day of Week',
                                            yaxis_title='Average Stock Level'
                                        )
                                        st.plotly_chart(fig_weekly)

                                    with season_col2:
                                        # Monthly patterns
                                        fig_monthly = go.Figure()
                                        fig_monthly.add_trace(go.Bar(
                                            x=['Jan',
                                                'Feb',
                                                'Mar',
                                                'Apr',
                                                'May',
                                                'Jun',
                                                'Jul',
                                                'Aug',
                                                'Sep',
                                                'Oct',
                                                'Nov',
                                                'Dec'],
                                            y=monthly_pattern.values,
                                            name='Average Stock Level'
                                        ))
                                        fig_monthly.update_layout(
                                            title='Monthly Pattern',
                                            xaxis_title='Month',
                                            yaxis_title='Average Stock Level'
                                        )
                                        st.plotly_chart(fig_monthly)

                                    # Advanced Time Series Analysis
                                    st.subheader(
                                        "🔄 Advanced Time Series Analysis")

                                    # Decomposition Analysis
                                    try:
                                        from statsmodels.tsa.seasonal import seasonal_decompose

                                        # Prepare time series data
                                        ts_data = historical_data.set_index('date')[
                                                    'stock_level']
                                        decomposition = seasonal_decompose(
                                            ts_data, period=30)

                                        # Plot decomposition components
                                        fig_decomp = go.Figure()
                                        fig_decomp.add_trace(
                                            go.Scatter(
                                                x=ts_data.index,
                                                y=decomposition.trend,
                                                name='Trend'))
                                        fig_decomp.add_trace(
                                            go.Scatter(
                                                x=ts_data.index,
                                                y=decomposition.seasonal,
                                                name='Seasonal'))
                                        fig_decomp.add_trace(
                                            go.Scatter(
                                                x=ts_data.index,
                                                y=decomposition.resid,
                                                name='Residual'))

                                        fig_decomp.update_layout(
                                            title='Time Series Decomposition',
                                            xaxis_title='Date',
                                            yaxis_title='Component Value'
                                        )
                                        st.plotly_chart(fig_decomp)
                                    except Exception as e:
                                        st.warning(
                                            f"Could not perform time series decomposition: {str(e)}")

                                    # Risk Assessment
                                    st.subheader("⚠️ Risk Assessment")

                                    # Calculate risk metrics
                                    risk_col1, risk_col2 = st.columns(2)

                                    with risk_col1:
                                        # Stock-out Risk
                                        current_stock = predictions[0]
                                        daily_usage = np.abs(
                                            np.diff(historical_data['stock_level'])).mean()
                                        days_until_stockout = current_stock / \
                                            daily_usage if daily_usage > 0 else float(
                                                'inf')

                                        stockout_risk = "High" if days_until_stockout < 7 else \
                                                      "Medium" if days_until_stockout < 14 else "Low"

                                        st.metric(
                                            "Stock-out Risk Level",
                                            stockout_risk,
                                            f"{days_until_stockout:.1f} days until stock-out"
                                        )

                                        # Demand Volatility
                                        volatility = historical_data['stock_level'].std(
                                        ) / historical_data['stock_level'].mean()
                                        volatility_risk = "High" if volatility > 0.5 else \
                                                        "Medium" if volatility > 0.2 else "Low"

                                        st.metric(
                                            "Demand Volatility",
                                            volatility_risk,
                                            f"{volatility:.1%} coefficient of variation"
                                        )

                                    with risk_col2:
                                        # Prediction Confidence
                                        confidence_interval = np.std(
                                            predictions) * 1.96
                                        prediction_range = f"±{confidence_interval:.1f} units"

                                        st.metric(
                                            "Prediction Confidence Interval",
                                            prediction_range,
                                            "95% confidence level"
                                        )

                                        # Supply Chain Risk
                                        lead_time_variation = np.random.normal(
                                            7, 2, 100)  # Simulated lead time data
                                        supply_risk = "High" if np.std(lead_time_variation) > 3 else \
                                                    "Medium" if np.std(
                                                        lead_time_variation) > 1 else "Low"

                                        st.metric(
                                            "Supply Chain Risk",
                                            supply_risk,
                                            f"Based on lead time variation"
                                        )

                                    # Risk Mitigation Recommendations
                                    st.subheader(
                                        "🛡️ Risk Mitigation Recommendations")

                                    recommendations = []

                                    # Stock-out risk recommendations
                                    if stockout_risk == "High":
                                        recommendations.append({
                                            "title": "Critical Stock Level Alert",
                                            "description": "Immediate reorder recommended. Consider expedited shipping options.",
                                            "priority": "High"
                                        })

                                    # Volatility recommendations
                                    if volatility_risk == "High":
                                        recommendations.append({
                                            "title": "High Demand Volatility",
                                            "description": "Implement safety stock adjustments and consider automated reordering.",
                                            "priority": "Medium"
                                        })

                                    # Supply chain recommendations
                                    if supply_risk == "High":
                                        recommendations.append({
                                            "title": "Supply Chain Risk Alert",
                                            "description": "Diversify suppliers and increase safety stock levels.",
                                            "priority": "High"
                                        })

                                    # Display recommendations
                                    for rec in recommendations:
                                        with st.expander(f"{rec['title']} ({rec['priority']} Priority)"):
                                            st.write(rec['description'])

                                    # Model Performance Metrics
                                    st.subheader("📊 Model Performance Metrics")

                                    metrics_col1, metrics_col2 = st.columns(2)

                                    with metrics_col1:
                                        # Calculate error metrics
                                        mse = np.mean(
                                            (historical_data['stock_level'].iloc[-len(predictions):].values - predictions) ** 2)
                                        rmse = np.sqrt(mse)
                                        mae = np.mean(np.abs(
                                            historical_data['stock_level'].iloc[-len(predictions):].values - predictions))

                                        st.metric(
                                            "Root Mean Square Error", f"{rmse:.2f}")
                                        st.metric(
                                            "Mean Absolute Error", f"{mae:.2f}")

                                    with metrics_col2:
                                        # Calculate accuracy metrics
                                        direction_accuracy = np.mean(np.sign(np.diff(predictions)) ==
                                                                   np.sign(np.diff(historical_data['stock_level'].iloc[-len(predictions):].values))) * 100

                                        st.metric(
                                            "Direction Accuracy", f"{direction_accuracy:.1f}%")
                                else:
                                    st.error("Failed to generate predictions")

                                # Remove progress bar after completion
                                progress_bar.empty()
                            else:
                                st.error("Failed to initialize LSTM model")
                        else:
                            st.error("Insufficient data for LSTM model")
                    except Exception as e:
                        st.error(f"Error in prediction process: {str(e)}")
                        progress_bar.empty()
                else:
                    st.error("No historical data available for predictions")

        # Demand Pattern Analysis
            st.subheader("Demand Pattern Analysis")
        analyze_demand_patterns(historical_data)

elif page == "Optimization":
    st.title("Inventory Optimization")

    # Get current inventory data
    inventory_data = db.get_current_stock_levels()

    # Optimization Parameters
    st.subheader("Optimization Settings")

    col1, col2 = st.columns(2)
    with col1:
        holding_cost = st.number_input("Holding Cost per Unit (%)", value=10, min_value=1, max_value=100)
        stockout_cost = st.number_input("Stockout Cost per Unit (₹)", value=2000, min_value=1)
            
    with col2:
        lead_time = st.number_input("Lead Time (Days)", value=7, min_value=1)
        service_level = st.number_input("Service Level (%)", value=95, min_value=50, max_value=100)
    
    # Calculate optimal stock levels
    z_score = np.abs(np.percentile(np.random.standard_normal(10000), service_level))
    
    if inventory_data.empty:
        st.warning("No inventory data available for optimization")
    else:
        optimization_results = []
        for index, item in inventory_data.iterrows():
            try:
                demand_std = item['stock_level'] * 0.2  # Example standard deviation
                safety_stock = z_score * demand_std * np.sqrt(lead_time/30)
                reorder_point = (item['stock_level']/30) * lead_time + safety_stock
                
                optimization_results.append({
                    'product_name': item.get('product_name', f'Product_{index}'),  # Fallback if name missing
                    'current_stock': item['stock_level'],
                    'safety_stock': round(safety_stock, 1),
                    'reorder_point': round(reorder_point, 1),
                    'optimal_order': round(max(0, reorder_point - item['stock_level']), 1)
                })
            except Exception as e:
                st.error(f"Error processing item {index}: {str(e)}")
                continue
        
        if optimization_results:
            optimization_df = pd.DataFrame(optimization_results)
            
            # Display optimization results
            st.subheader("Optimization Results")
            st.dataframe(
                optimization_df,
                column_config={
                    'product_name': 'Product',
                    'current_stock': st.column_config.NumberColumn('Current Stock', format='%d'),
                    'safety_stock': st.column_config.NumberColumn('Safety Stock', format='%.1f'),
                    'reorder_point': st.column_config.NumberColumn('Reorder Point', format='%.1f'),
                    'optimal_order': st.column_config.NumberColumn('Recommended Order', format='%.1f')
                }
            )
            
            # Visualization of optimization results
            st.subheader("Stock Level Analysis")
            try:
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    name='Current Stock',
                    x=optimization_df['product_name'],
                    y=optimization_df['current_stock']
                ))
                
                fig.add_trace(go.Bar(
                    name='Safety Stock',
                    x=optimization_df['product_name'],
                    y=optimization_df['safety_stock']
                ))
                
                fig.add_trace(go.Bar(
                    name='Reorder Point',
                    x=optimization_df['product_name'],
                    y=optimization_df['reorder_point']
                ))
                
                fig.update_layout(
                    barmode='group',
                    title='Stock Levels and Optimization Targets',
                    xaxis_title='Product',
                    yaxis_title='Units'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")
        else:
            st.warning("No valid data available for optimization")

elif page == "Settings":
    st.title("AI Model Settings")
    
    # Currency Settings
    st.subheader("🌐 Currency Settings")
    selected_currency = st.selectbox(
        "Select Currency",
        options=list(CURRENCY_SYMBOLS.keys()),
        index=list(CURRENCY_SYMBOLS.keys()).index(st.session_state.currency)
    )
    
    if selected_currency != st.session_state.currency:
        st.session_state.currency = selected_currency
        st.session_state.currency_symbol = CURRENCY_SYMBOLS[selected_currency]
        st.success(f"Currency changed to {selected_currency}")
    
    # Sample Data Generation
    st.subheader("📊 Sample Data Generation")
    with st.expander("Generate Sample Data", expanded=True):
        st.info("""
        **Data Requirements for AI Predictions:**
        - Minimum 60 days of historical data is required for LSTM model training
        - Multiple sales per day are generated to ensure sufficient data points
        - Data includes stock levels, prices, and transaction timestamps
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            num_products = st.number_input("Number of Products per Category", min_value=1, max_value=10, value=3)
            min_daily_sales = st.number_input("Minimum Sales per Day", min_value=2, max_value=10, value=2, 
                help="At least 2 sales per day recommended for sufficient data points")
            max_daily_sales = st.number_input("Maximum Sales per Day", min_value=2, max_value=20, value=5)
        with col2:
            days_of_history = st.number_input("Days of History", min_value=60, max_value=365, value=90, 
                help="Minimum 60 days required for LSTM model training")
            restock_probability = st.slider("Daily Restock Probability", min_value=0.0, max_value=1.0, value=0.2,
                help="Probability of restocking products each day")
        
        if st.button("Generate New Sample Data", type="primary"):
            try:
                with st.spinner("Generating comprehensive sample data..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    success, message = generate_sample_data(
                        db=db,
                        num_products=num_products,
                        min_daily_sales=min_daily_sales,
                        max_daily_sales=max_daily_sales,
                        days_of_history=days_of_history,
                        restock_probability=restock_probability
                    )
                    
                    if success:
                        st.success(f"""Sample data generated successfully!
                        - {num_products} products per category
                        - {days_of_history} days of historical data
                        - {min_daily_sales}-{max_daily_sales} sales per day
                        - Daily restock probability: {restock_probability * 100}%
                        
                        The data has been structured to support LSTM model training with:
                        - Consistent daily records
                        - Proper datetime indexing
                        - Sufficient historical depth ({days_of_history} days)
                        """)
                    else:
                        st.error(f"Failed to generate sample data: {message}")
                    
                    progress_bar.empty()
                    status_text.empty()
                    
            except Exception as e:
                st.error(f"Error generating sample data: {str(e)}")
    
    # Existing Settings UI
    st.subheader("Model Parameters")
    
    # LSTM Model Settings
    st.write("LSTM Model Configuration")
    lstm_lookback = st.slider("Lookback Period (Days)", 7, 60, 30)
    lstm_layers = st.slider("LSTM Layers", 1, 4, 2)
    lstm_units = st.slider("Units per Layer", 10, 100, 50)
    
    # Anomaly Detection Settings
    st.write("Anomaly Detection Configuration")
    contamination = st.slider("Contamination Factor", 0.01, 0.5, 0.1)
    
    # Prophet Model Settings
    st.write("Prophet Model Configuration")
    yearly_seasonality = st.checkbox("Yearly Seasonality", value=True)
    weekly_seasonality = st.checkbox("Weekly Seasonality", value=True)
    daily_seasonality = st.checkbox("Daily Seasonality", value=False)
    
    if st.button("Retrain Models"):
        with st.spinner("Retraining AI models..."):
            # Clear model caches
            st.cache_resource.clear()
            st.cache_data.clear()
            st.success("Models cache cleared and ready for retraining!")

elif page == "Inventory Management":
    st.title("Inventory Management")
    
    # Initialize inventory data with proper checks
    try:
        inventory_data = db.get_current_stock_levels()
        if inventory_data is None:
            inventory_data = pd.DataFrame(columns=['product_name', 'category', 'stock_level', 'price'])
    except Exception as e:
        st.error(f"Error loading inventory data: {str(e)}")
        inventory_data = pd.DataFrame(columns=['product_name', 'category', 'stock_level', 'price'])
    
    # Tabs for different operations
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["View Inventory", "Add/Edit Item", "Bulk Upload", "Download Template", "Analysis"])
    
    with tab1:
        st.subheader("Current Inventory")
        
        if not inventory_data.empty:
            try:
                # Ensure all required columns exist
                required_columns = ['product_name', 'category', 'stock_level', 'price']
                for col in required_columns:
                    if col not in inventory_data.columns:
                        inventory_data[col] = None
                
                # Add action buttons for each row
                inventory_data['Actions'] = None
                
                # Get category list with fallback
                try:
                    categories = db.get_category_list()
                except:
                    categories = ["Electronics", "Mobile Accessories", "Home Appliances", "Computer Parts", "Audio Devices"]
                
                edited_df = st.data_editor(
                    inventory_data,
                    column_config={
                        "product_name": st.column_config.TextColumn(
                            "Product Name",
                            required=True
                        ),
                        "category": st.column_config.SelectboxColumn(
                            "Category",
                            options=categories,
                            required=True
                        ),
                        "stock_level": st.column_config.NumberColumn(
                            "Stock Level",
                            min_value=0,
                            format="%d",
                            required=True
                        ),
                        "price": st.column_config.NumberColumn(
                            "Price (₹)",
                            min_value=0.0,
                            format="%.2f",
                            required=True
                        )
                    },
                    hide_index=True,
                    num_rows="dynamic"
                )
                
                if st.button("Save Changes"):
                    try:
                        if edited_df is not None and not edited_df.empty:
                            # Validate data before saving
                            edited_df['stock_level'] = pd.to_numeric(edited_df['stock_level'], errors='coerce')
                            edited_df['price'] = pd.to_numeric(edited_df['price'], errors='coerce')
                            
                            # Remove rows with invalid data
                            edited_df = edited_df.dropna(subset=['product_name', 'category', 'stock_level', 'price'])
                            
                            # Update database with edited data
                            for _, row in edited_df.iterrows():
                                db.save_product(row.to_dict())
                            st.success("Inventory updated successfully!")
                        else:
                            st.warning("No valid data to save.")
                    except Exception as e:
                        st.error(f"Error updating inventory: {str(e)}")
            except Exception as e:
                st.error(f"Error displaying inventory editor: {str(e)}")
        else:
            st.info("No inventory data available. Add items using the 'Add/Edit Item' tab.")
    
    with tab2:
        st.subheader("Add/Edit Inventory Item")
        
        # Form for adding/editing single item
        with st.form("inventory_form"):
            product_name = st.text_input("Product Name")
            category = st.selectbox("Category", db.get_category_list())
            stock_level = st.number_input("Stock Level", min_value=0)
            price = st.number_input("Price (₹)", min_value=0.0, format="%.2f")
            notes = st.text_area("Notes")
            
            submitted = st.form_submit_button("Save Item")
            if submitted:
                try:
                    item_data = {
                        "product_name": product_name,
                        "category": category,
                        "stock_level": stock_level,
                        "price": price,
                        "notes": notes,
                        "last_updated": datetime.now()
                    }
                    db.save_product(item_data)
                    st.success("Item saved successfully!")
                except Exception as e:
                    st.error(f"Error saving item: {str(e)}")
    
    with tab3:
        st.subheader("Bulk Upload")
        
        uploaded_file = st.file_uploader("Upload Inventory Data (Excel/CSV)", type=['xlsx', 'csv'])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.write("Preview of uploaded data:")
                st.dataframe(df.head())
                
                if st.button("Process Bulk Upload"):
                    # Validate required columns
                    required_cols = ['product_name', 'category', 'stock_level', 'price']
                    if all(col in df.columns for col in required_cols):
                        # Convert DataFrame to list of dictionaries and add timestamp
                        records = df.to_dict('records')
                        for record in records:
                            record['last_updated'] = datetime.now()
                            db.save_product(record)
                        st.success(f"Successfully uploaded {len(records)} items!")
                    else:
                        st.error("Missing required columns. Please use the template.")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    with tab4:
        st.subheader("Download Template")
        
        # Create template DataFrame
        template_df = pd.DataFrame(columns=[
            'product_name',
            'category',
            'stock_level',
            'price',
            'supplier',
            'notes'
        ])
        
        # Add sample row
        template_df.loc[0] = [
            'Sample Product',
            'Electronics',
            100,
            15999.99,  # Changed to typical Indian price point
            'Sample Supplier',
            'Sample product notes'
        ]
        
        # Create Excel file in memory
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            template_df.to_excel(writer, index=False, sheet_name='Inventory Template')
            # Get workbook and worksheet objects
            workbook = writer.book
            worksheet = writer.sheets['Inventory Template']
            
            # Add formats
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#0066cc',
                'font_color': 'white'
            })
            
            # Apply formats
            for col_num, value in enumerate(template_df.columns.values):
                worksheet.write(0, col_num, value, header_format)
                worksheet.set_column(col_num, col_num, 15)
        
        # Offer download button
        st.download_button(
            label="Download Template",
            data=buffer.getvalue(),
            file_name="inventory_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        st.markdown("""
        ### Template Instructions
        1. **Product Name**: Unique identifier for the product
        2. **Category**: Product category (e.g., Electronics, Accessories)
        3. **Stock Level**: Current quantity in stock
        4. **Price**: Product price in INR
        5. **Supplier**: Product supplier name
        6. **Notes**: Additional product information
        
        **Note**: Product Name, Category, Stock Level, and Price are required fields.
        """)
    
    with tab5:
        st.subheader("Inventory Analysis")
        
        if not inventory_data.empty:
            try:
                # Ensure category exists, use 'Uncategorized' as fallback
                if 'category' not in inventory_data.columns:
                    inventory_data['category'] = 'Uncategorized'
                
                # Category-wise analysis
                st.write("### Category Distribution")
                category_counts = inventory_data.groupby('category').agg({
                    'product_name': 'count',
                    'stock_level': 'sum',
                    'price': 'mean'
                }).round(2)
                
                category_counts.columns = ['Number of Products', 'Total Stock', 'Average Price']
                st.dataframe(category_counts)
                
                # Visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    try:
                        # Category-wise stock distribution
                        fig_stock = px.pie(
                            inventory_data,
                            values='stock_level',
                            names='category',
                            title='Stock Distribution by Category'
                        )
                        st.plotly_chart(fig_stock)
                    except Exception as e:
                        st.error(f"Error creating stock distribution chart: {str(e)}")
                
                with col2:
                    try:
                        # Price range distribution
                        fig_price = px.histogram(
                            inventory_data,
                            x='price',
                            nbins=20,
                            title='Price Distribution'
                        )
                        st.plotly_chart(fig_price)
                    except Exception as e:
                        st.error(f"Error creating price distribution chart: {str(e)}")
                
                # Stock level analysis
                st.write("### Stock Level Analysis")
                low_stock = inventory_data[inventory_data['stock_level'] < 50]
                if not low_stock.empty:
                    st.warning("Products with Low Stock (< 50 units):")
                    st.dataframe(low_stock[['product_name', 'category', 'stock_level']])
                
                # Value analysis
                try:
                    total_value = (inventory_data['stock_level'] * inventory_data['price']).sum()
                    avg_value = (inventory_data['stock_level'] * inventory_data['price']).mean()
                    
                    col3, col4 = st.columns(2)
                    with col3:
                        st.metric("Total Inventory Value", f"{st.session_state.currency_symbol}{total_value:,.2f}")
                    with col4:
                        st.metric("Average Product Value", f"{st.session_state.currency_symbol}{avg_value:,.2f}")
                except Exception as e:
                    st.error(f"Error calculating inventory values: {str(e)}")
                
                # Stock vs Price correlation
                try:
                    fig_correlation = px.scatter(
                        inventory_data,
                        x='price',
                        y='stock_level',
                        color='category',
                        title='Stock Level vs Price by Category',
                        labels={'price': 'Price', 'stock_level': 'Stock Level'}
                    )
                    st.plotly_chart(fig_correlation)
                except Exception as e:
                    st.error(f"Error creating correlation chart: {str(e)}")
            except Exception as e:
                st.error(f"Error in inventory analysis: {str(e)}")
                st.info("Please ensure all required data (category, stock level, price) is available")
        else:
            st.info("No inventory data available for analysis")

# Add any remaining necessary code for inventory management functionality 

app = FastAPI(
    title="RetailSense AI Inventory Management",
    description="AI-powered inventory management system for Indian retail market",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize inventory manager
inventory_manager = InventoryManager(os.getenv("MONGODB_URI"))

# Pydantic models for request/response validation
class ProductBase(BaseModel):
    name: str
    category: str
    price: float
    quantity: int
    min_stock: int
    max_stock: int
    supplier_id: Optional[str]
    description: Optional[str]

class ProductUpdate(BaseModel):
    name: Optional[str]
    category: Optional[str]
    price: Optional[float]
    quantity: Optional[int]
    min_stock: Optional[int]
    max_stock: Optional[int]
    supplier_id: Optional[str]
    description: Optional[str]

class TransactionBase(BaseModel):
    product_id: str
    type: str  # 'purchase' or 'sale'
    quantity: int
    unit_price: float
    total_amount: float
    customer_id: Optional[str]
    supplier_id: Optional[str]
    notes: Optional[str]

class DateRange(BaseModel):
    start_date: datetime
    end_date: datetime

@app.post("/products/", response_model=Dict)
async def create_product(product: ProductBase):
    """Create a new product in inventory"""
    try:
        product_id = inventory_manager.add_product(product.dict())
        return {"product_id": product_id, "message": "Product created successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.put("/products/{product_id}", response_model=Dict)
async def update_product(product_id: str, updates: ProductUpdate):
    """Update product information"""
    try:
        success = inventory_manager.update_product(product_id, updates.dict(exclude_unset=True))
        if not success:
            raise HTTPException(status_code=404, detail="Product not found")
        return {"message": "Product updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/products/{product_id}", response_model=Dict)
async def get_product(product_id: str):
    """Get product information with AI insights"""
    try:
        result = inventory_manager.get_inventory_status(product_id)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/transactions/", response_model=Dict)
async def record_transaction(transaction: TransactionBase):
    """Record an inventory transaction"""
    try:
        transaction_id = inventory_manager.record_transaction(transaction.dict())
        return {"transaction_id": transaction_id, "message": "Transaction recorded successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/inventory/status", response_model=Dict)
async def get_inventory_status():
    """Get overall inventory status with AI insights"""
    try:
        return inventory_manager.get_inventory_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/reports/generate", response_model=Dict)
async def generate_report(date_range: DateRange):
    """Generate inventory reports with AI insights"""
    try:
        return inventory_manager.generate_reports(
            date_range.start_date,
            date_range.end_date
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
