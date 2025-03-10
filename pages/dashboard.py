import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from models.sales import SalesManager
from models.inventory_manager import InventoryManager
from database import MongoDB
import pandas as pd
import logging
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
import warnings
import random
import time
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

def get_db_connection():
    """Initialize database connection with retries"""
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            db = MongoDB()
            if not db.validate_connection():
                raise ConnectionError("Database connection validation failed")
            logger.info("Successfully connected to database")
            return db
        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                logger.error(f"Failed to connect to database after {max_retries} attempts: {str(e)}")
                return None
            logger.warning(f"Connection attempt {attempt + 1}/{max_retries} failed: {str(e)}")
            time.sleep(retry_delay)
    return None

def calculate_stock_turnover(db):
    """Calculate stock turnover rate for products"""
    try:
        if not db:
            raise ValueError("Database connection not available")
            
        # Get sales data for the last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # Get sales data
        sales_data = db.get_sales_data()
        if not sales_data:
            return pd.DataFrame(columns=['product_name', 'category', 'sales_quantity', 'avg_inventory', 'turnover_rate'])
        
        sales_df = pd.DataFrame(sales_data)
        sales_df['timestamp'] = pd.to_datetime(sales_df['timestamp'])
        sales_df = sales_df[sales_df['timestamp'].between(start_date, end_date)]
        
        # Get current inventory data
        inventory_data = db.get_inventory_status()
        if not inventory_data:
            return pd.DataFrame(columns=['product_name', 'category', 'sales_quantity', 'avg_inventory', 'turnover_rate'])
        
        inventory_df = pd.DataFrame(inventory_data)
        
        # Calculate sales quantity per product
        sales_by_product = []
        for _, sale in sales_df.iterrows():
            for item in sale['items']:
                sales_by_product.append({
                    'product_name': item['name'],
                    'quantity': item['quantity']
                })
        
        sales_summary = pd.DataFrame(sales_by_product)
        if not sales_summary.empty:
            sales_summary = sales_summary.groupby('product_name')['quantity'].sum().reset_index()
        
        # Merge sales and inventory data
        turnover_data = []
        for _, product in inventory_df.iterrows():
            sales_qty = sales_summary[sales_summary['product_name'] == product['name']]['quantity'].sum() if not sales_summary.empty else 0
            avg_inventory = product['quantity']
            
            # Calculate turnover rate (sales / average inventory)
            turnover_rate = sales_qty / avg_inventory if avg_inventory > 0 else 0
            
            turnover_data.append({
                'product_name': product['name'],
                'category': product['category'],
                'sales_quantity': sales_qty,
                'avg_inventory': avg_inventory,
                'turnover_rate': turnover_rate
            })
        
        # Create DataFrame and sort by turnover rate
        turnover_df = pd.DataFrame(turnover_data)
        turnover_df = turnover_df.sort_values('turnover_rate', ascending=False)
        
        return turnover_df
        
    except Exception as e:
        logger.error(f"Error calculating stock turnover: {str(e)}")
        return pd.DataFrame(columns=['product_name', 'category', 'sales_quantity', 'avg_inventory', 'turnover_rate'])

def render_sales_entry_form(sales_manager, inventory_manager):
    """Render the sales entry form in a sidebar"""
    with st.sidebar.expander("📝 Add New Sale", expanded=False):
        st.markdown("### Add New Sale")
        st.markdown("_Enter details for a new sales transaction_")
        
        # Get product list for selection
        products = inventory_manager.get_all_products()
        if not products:
            st.error("No products available in inventory")
            return
        
        with st.form("sales_entry_form"):
            # Customer Information
            st.subheader("Customer Information")
            customer_name = st.text_input("Customer Name", help="Enter customer's full name")
            col1, col2 = st.columns(2)
            with col1:
                customer_phone = st.text_input("Phone Number", help="Enter customer's contact number")
            with col2:
                customer_email = st.text_input("Email (optional)", help="Enter customer's email address")
            
            # Product Selection
            st.subheader("Product Selection")
            num_items = st.number_input("Number of Items", min_value=1, max_value=10, value=1,
                                      help="Select number of different products")
            
            items = []
            total_amount = 0
            
            for i in range(int(num_items)):
                st.markdown(f"**Item {i + 1}**")
                col1, col2 = st.columns(2)
                
                with col1:
                    product = st.selectbox(
                        "Select Product",
                        options=products,
                        format_func=lambda x: f"{x['name']} (₹{x.get('price', 0)})",
                        key=f"product_{i}",
                        help="Choose a product from inventory"
                    )
                
                with col2:
                    max_qty = int(product.get('quantity', 0)) if product else 0
                    quantity = st.number_input(
                        "Quantity",
                        min_value=1,
                        max_value=max_qty,
                        value=1,
                        key=f"quantity_{i}",
                        help=f"Available stock: {max_qty}"
                    )
                
                if product:
                    price = float(product.get('price', 0))
                    amount = price * quantity
                    total_amount += amount
                    items.append({
                        'product_id': str(product.get('_id')),
                        'name': product.get('name'),
                        'quantity': quantity,
                        'price': price,
                        'amount': amount
                    })
            
            st.markdown(f"**Total Amount: ₹{total_amount:,.2f}**")
            
            # Payment Information
            st.subheader("Payment Information")
            payment_method = st.selectbox(
                "Payment Method",
                options=["Cash", "Card", "UPI", "Other"],
                help="Select payment method used"
            )
            
            # Additional Notes
            notes = st.text_area("Notes (optional)", help="Add any additional notes about the sale")
            
            submit = st.form_submit_button("Complete Sale")
            
            if submit:
                if not customer_name:
                    st.error("Please enter customer name")
                    return
                
                if not customer_phone:
                    st.error("Please enter customer phone number")
                    return
                
                if not items:
                    st.error("Please add at least one item")
                    return
                
                # Create sale record with all required fields
                sale_data = {
                    'customer': {
                        'name': customer_name,
                        'phone': customer_phone,
                        'email': customer_email if customer_email else None
                    },
                    'items': items,
                    'total_amount': total_amount,
                    'payment_method': payment_method,
                    'timestamp': datetime.utcnow(),
                    'status': 'completed',
                    'notes': notes if notes else None,
                    'created_at': datetime.utcnow(),
                    'updated_at': datetime.utcnow()
                }
                
                try:
                    # Validate sale data before saving
                    if not all(key in sale_data for key in ['customer', 'items', 'total_amount', 'payment_method']):
                        st.error("Missing required fields in sale data")
                        return
                    
                    if not sale_data['items']:
                        st.error("No items in sale data")
                        return
                    
                    for item in sale_data['items']:
                        if not all(key in item for key in ['product_id', 'quantity', 'price']):
                            st.error(f"Missing required fields in item: {item.get('name', 'Unknown')}")
                            return
                    
                    # Add sale and update inventory
                    sales_manager.save_sales_data(sale_data)
                    st.success("Sale completed successfully!")
                    
                    # Show receipt
                    st.markdown("### Receipt")
                    receipt = f"""
                    **Date:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}
                    
                    **Customer Information:**
                    - Name: {customer_name}
                    - Phone: {customer_phone}
                    {f'- Email: {customer_email}' if customer_email else ''}
                    
                    **Items:**
                    """
                    
                    for item in items:
                        receipt += f"\n- {item['name']} x {item['quantity']} = ₹{item['amount']:,.2f}"
                    
                    receipt += f"""
                    
                    **Total Amount:** ₹{total_amount:,.2f}
                    **Payment Method:** {payment_method}
                    {f'**Notes:** {notes}' if notes else ''}
                    
                    Thank you for your purchase!
                    """
                    
                    st.markdown(receipt)
                    
                    # Clear form (by rerunning the app)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error completing sale: {str(e)}")
                    logger.error(f"Error in sales entry: {str(e)}")
                    return

def render_dashboard():
    """Render the main dashboard"""
    # Initialize database connection
    db = get_db_connection()
    if not db:
        st.error("""
        Failed to connect to database.
        
        Please check:
        1. MongoDB connection string in .env file
        2. MongoDB service is running
        3. Network connectivity
        4. Database permissions
        """)
        return

    # Set page config with error handling
    try:
        st.set_page_config(
            page_title="RetailSense Dashboard",
            page_icon="📊",
            layout="wide"
        )
    except Exception as e:
        logger.error(f"Failed to set page config: {str(e)}")
        st.error("Failed to initialize dashboard layout")

    # Add custom CSS with error handling
    try:
        st.markdown("""
            <style>
            .stButton>button {
                width: 100%;
                margin-top: 10px;
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                transition: background-color 0.3s;
            }
            .stButton>button:hover {
                background-color: #45a049;
            }
            .stMetric {
                background-color: #f0f2f6;
                padding: 15px;
                border-radius: 5px;
                margin: 5px 0;
            }
            .stMetric:hover {
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            .error-message {
                color: #dc3545;
                padding: 10px;
                border-radius: 5px;
                background-color: #f8d7da;
                margin: 10px 0;
            }
            .success-message {
                color: #28a745;
                padding: 10px;
                border-radius: 5px;
                background-color: #d4edda;
                margin: 10px 0;
            }
            </style>
        """, unsafe_allow_html=True)
    except Exception as e:
        logger.error(f"Failed to add custom CSS: {str(e)}")

    # Get data for dashboard with proper error handling
    try:
        # Get sales data with validation
        sales_data = db.get_sales_data()
        if sales_data:
            sales_df = pd.DataFrame(sales_data)
            required_columns = ['timestamp', 'total_amount']
            if not all(col in sales_df.columns for col in required_columns):
                raise ValueError(f"Missing required columns in sales data: {required_columns}")
            
            sales_df['date'] = pd.to_datetime(sales_df['timestamp']).dt.date
            total_sales = sales_df['total_amount'].sum()
            total_orders = len(sales_df)
            avg_order_value = total_sales / total_orders if total_orders > 0 else 0
        else:
            sales_df = pd.DataFrame(columns=['date', 'amount'])
            total_sales = 0
            total_orders = 0
            avg_order_value = 0
            st.info("No sales data available")

        # Get inventory data with validation
        inventory_data = db.get_inventory_status()
        if inventory_data:
            inventory_df = pd.DataFrame(inventory_data)
            required_columns = ['name', 'category', 'quantity', 'price', 'min_stock']
            if not all(col in inventory_df.columns for col in required_columns):
                raise ValueError(f"Missing required columns in inventory data: {required_columns}")
            
            total_products = len(inventory_df)
            total_inventory_value = (inventory_df['quantity'] * inventory_df['price']).sum()
            low_stock_count = len(inventory_df[inventory_df['quantity'] <= inventory_df['min_stock']])
            out_of_stock_count = len(inventory_df[inventory_df['quantity'] == 0])
            
            # Calculate category sales with validation
            if 'category' in sales_df.columns:
                category_sales = sales_df.groupby('category')['total_amount'].sum().reset_index()
            else:
                category_sales = pd.DataFrame(columns=['category', 'total_amount'])
            
            # Calculate low stock items with validation
            low_stock_df = inventory_df[inventory_df['quantity'] <= inventory_df['min_stock']].copy()
            if not low_stock_df.empty:
                low_stock_df['stock_percentage'] = (low_stock_df['quantity'] / low_stock_df['min_stock']) * 100
        else:
            inventory_df = pd.DataFrame(columns=['name', 'category', 'quantity', 'price', 'min_stock', 'status'])
            total_products = 0
            total_inventory_value = 0
            low_stock_count = 0
            out_of_stock_count = 0
            category_sales = pd.DataFrame(columns=['category', 'amount'])
            low_stock_df = pd.DataFrame()
            st.info("No inventory data available")

        # Calculate stock turnover
        turnover_df = calculate_stock_turnover(db)

        # Get AI insights
        anomalies = db.detect_anomalies()
        recommendations = db.get_stock_recommendations()

        # Render dashboard components
        st.title("RetailSense Dashboard")
        
        # Summary Metrics
        st.header("Summary Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Sales", f"₹{total_sales:,.2f}")
        with col2:
            st.metric("Total Orders", f"{total_orders:,}")
        with col3:
            st.metric("Average Order Value", f"₹{avg_order_value:,.2f}")
        with col4:
            st.metric("Total Products", f"{total_products:,}")
        
        # Inventory Status
        st.header("Inventory Status")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Products", f"{total_products:,}")
        with col2:
            st.metric("Total Value", f"₹{total_inventory_value:,.2f}")
        with col3:
            st.metric("Low Stock Items", f"{low_stock_count:,}")
        with col4:
            st.metric("Out of Stock", f"{out_of_stock_count:,}")
        
        # AI Insights
        st.header("AI Insights")
        
        # Anomaly Detection
        st.subheader("Anomaly Detection")
        if anomalies:
            for anomaly in anomalies:
                st.warning(f"**{anomaly['product_name']}**: {anomaly['reason']}")
        else:
            st.success("No anomalies detected")
        
        # Stock Recommendations
        st.subheader("Stock Recommendations")
        if recommendations:
            for rec in recommendations:
                st.info(f"**{rec['product_name']}**: {rec['recommendation']} (Priority: {rec['priority']})")
        else:
            st.success("No stock recommendations needed")

    except Exception as e:
        st.error(f"""
        Error loading dashboard data:
        
        {str(e)}
        
        Please try:
        1. Refreshing the page
        2. Checking database connectivity
        3. Verifying data integrity
        """)
        logger.error(f"Error loading dashboard data: {str(e)}")
        return

    # Sidebar
    with st.sidebar:
        st.title("RetailSense")
        st.markdown("---")
        
        # Date Range Selector
        st.subheader("Date Range")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
        with col2:
            end_date = st.date_input("End Date", datetime.now())
        
        # Category Filter
        st.subheader("Category Filter")
        categories = ["All"] + db.get_category_list()
        selected_category = st.selectbox("Select Category", categories)
        
        # Sample Data Generation
        st.markdown("---")
        st.subheader("Sample Data")
        if st.button("🔄 Generate Sample Data", key="generate_sample_data"):
            with st.spinner("Generating sample data..."):
                try:
                    if db.generate_sample_data():
                        st.success("Sample data generated successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to generate sample data. Please check the logs.")
                except Exception as e:
                    st.error(f"Error generating sample data: {str(e)}")
        
        # AI Model Settings
        st.markdown("---")
        st.subheader("AI Model Settings")
        
        # Model Configuration
        model_type = st.selectbox(
            "Select Model Type",
            ["LSTM", "Prophet", "ARIMA"],
            key="model_type"
        )
        
        # Advanced Settings
        st.markdown("### Advanced Settings")
        
        # LSTM Settings
        if model_type == "LSTM":
            lstm_layers = st.slider("Number of LSTM Layers", 1, 3, 2)
            lstm_units = st.slider("LSTM Units per Layer", 32, 256, 64)
            dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, 0.2)
        
        # Prophet Settings
        elif model_type == "Prophet":
            changepoint_prior_scale = st.slider("Changepoint Prior Scale", 0.001, 0.5, 0.05)
            seasonality_prior_scale = st.slider("Seasonality Prior Scale", 0.01, 10.0, 1.0)
            holidays_prior_scale = st.slider("Holidays Prior Scale", 0.01, 10.0, 1.0)
        
        # ARIMA Settings
        else:  # ARIMA
            p = st.slider("AR Order (p)", 0, 5, 1)
            d = st.slider("Difference Order (d)", 0, 3, 1)
            q = st.slider("MA Order (q)", 0, 5, 1)
        
        # Common Settings
        forecast_horizon = st.slider("Forecast Horizon (days)", 7, 90, 30)
        confidence_interval = st.slider("Confidence Interval (%)", 80, 99, 95)
        
        # Model Training
        st.markdown("### Model Training")
        if st.button("Train Model", key="train_model"):
            with st.spinner("Training model..."):
                try:
                    # Get historical data
                    historical_data = db.get_historical_stock_data()
                    if historical_data.empty:
                        st.error("Insufficient historical data for training")
                        return
                    
                    # Train model based on selected type
                    if model_type == "LSTM":
                        model = train_lstm_model(historical_data, lstm_layers, lstm_units, dropout_rate)
                    elif model_type == "Prophet":
                        model = train_prophet_model(historical_data, changepoint_prior_scale, seasonality_prior_scale, holidays_prior_scale)
                    else:  # ARIMA
                        model = train_arima_model(historical_data, p, d, q)
                    
                    st.success("Model trained successfully!")
                except Exception as e:
                    st.error(f"Error training model: {str(e)}")
        
        # Model Evaluation
        st.markdown("### Model Evaluation")
        if st.button("Evaluate Model", key="evaluate_model"):
            with st.spinner("Evaluating model..."):
                try:
                    # Get test data
                    test_data = db.get_historical_stock_data(days=30)
                    if test_data.empty:
                        st.error("Insufficient test data")
                        return
                    
                    # Evaluate model
                    metrics = evaluate_model(model, test_data)
                    
                    # Display metrics
                    st.markdown("#### Model Performance Metrics")
                    for metric, value in metrics.items():
                        st.metric(metric, f"{value:.2f}")
                except Exception as e:
                    st.error(f"Error evaluating model: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit")

if __name__ == "__main__":
    render_dashboard() 