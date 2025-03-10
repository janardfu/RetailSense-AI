from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from datetime import datetime
from database import MongoDB
import io
import plotly.express as px
import numpy as np

# Page configuration
st.set_page_config(
    page_title="RetailSense Dashboard",
    page_icon="📊",
    layout="wide"
)

# Initialize MongoDB connection
db = MongoDB()

load_dotenv()  # Load environment variables from .env file

# Sidebar navigation
st.sidebar.title("RetailSense")
page = st.sidebar.selectbox(
    "Navigation",
    ["Dashboard", "Inventory", "Analytics", "Products", "Categories"]
)

# Main content
if page == "Dashboard":
    st.title("Dashboard")
    
    # Get current data
    sales_data = db.get_sales_data()
    current_stock = db.get_current_stock_levels()
    
    # KPI metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        try:
            total_sales = sales_data[sales_data['date'] == sales_data['date'].max()]['sales'].sum()
            st.metric(
                label="Today's Sales",
                value=f"${total_sales:,.2f}",
                delta=f"{np.random.randint(-10, 20)}%"
            )
        except Exception as e:
            st.error(f"Error calculating total sales: {str(e)}")
            st.metric(label="Today's Sales", value="N/A")
    
    with col2:
        try:
            if 'product_name' in current_stock.columns:
                total_products = len(current_stock['product_name'].unique())
            else:
                total_products = 0
            st.metric(
                label="Total Products",
                value=total_products,
                delta=None
            )
        except Exception as e:
            st.error(f"Error calculating total products: {str(e)}")
            st.metric(label="Total Products", value="N/A")
    
    with col3:
        try:
            if 'stock_level' in current_stock.columns:
                avg_stock = current_stock['stock_level'].mean()
                st.metric(
                    label="Average Stock Level",
                    value=f"{avg_stock:.0f}",
                    delta=None
                )
            else:
                st.metric(label="Average Stock Level", value="N/A")
        except Exception as e:
            st.error(f"Error calculating average stock: {str(e)}")
            st.metric(label="Average Stock Level", value="N/A")
    
    # Sales Trend
    st.subheader("Sales Trend")
    daily_sales = sales_data.groupby('date')['sales'].sum().reset_index()
    fig_sales = px.line(
        daily_sales,
        x='date',
        y='sales',
        title='Daily Sales Trend'
    )
    st.plotly_chart(fig_sales, use_container_width=True)
    
    # Product Performance
    st.subheader("Product Performance")
    product_sales = sales_data.groupby('product_name')['sales'].sum().reset_index()
    fig_products = px.bar(
        product_sales,
        x='product_name',
        y='sales',
        title='Sales by Product'
    )
    st.plotly_chart(fig_products, use_container_width=True)
    
    # Stock Alerts
    st.subheader("Stock Alerts")
    recommendations = db.get_stock_recommendations()
    for rec in recommendations:
        if "Low stock" in rec['recommendation']:
            st.warning(f"🔸 {rec['product_name']}: {rec['recommendation']}")
        else:
            st.info(f"🔹 {rec['product_name']}: {rec['recommendation']}")

elif page == "Inventory":
    st.title("Inventory Management")
    
    # Create tabs for different inventory operations
    inv_tab1, inv_tab2, inv_tab3 = st.tabs(["Current Stock", "Add Entry", "Bulk Upload"])
    
    with inv_tab1:
        # Get current stock levels
        stock_data = db.get_current_stock_levels()
        
        # Display inventory table
        st.dataframe(
            stock_data,
            column_config={
                "product_name": "Product Name",
                "stock_level": st.column_config.NumberColumn(
                    "Stock Level",
                    help="Current stock level",
                    format="%d"
                ),
                "price": st.column_config.NumberColumn(
                    "Price",
                    help="Current price",
                    format="$%.2f"
                )
            },
            use_container_width=True
        )
        
        # Stock Level Visualization
        st.subheader("Stock Levels by Product")
        fig_stock = px.bar(
            stock_data,
            x='product_name',
            y='stock_level',
            title='Current Stock Levels'
        )
        st.plotly_chart(fig_stock, use_container_width=True)
    
    with inv_tab2:
        st.subheader("Add New Inventory Entry")
        with st.form("add_inventory_entry"):
            # Get list of existing products and categories
            products = db.get_product_list()
            categories = db.get_category_list()
            
            # Form fields
            product_name = st.selectbox("Product", products) if products else st.text_input("Product Name")
            category = st.selectbox("Category", categories) if categories else st.text_input("Category")
            quantity = st.number_input("Quantity", min_value=0)
            unit_price = st.number_input("Unit Price", min_value=0.0, format="%.2f")
            supplier = st.text_input("Supplier")
            purchase_date = st.date_input("Purchase Date")
            expiry_date = st.date_input("Expiry Date")
            batch_number = st.text_input("Batch Number")
            notes = st.text_area("Notes")
            
            if st.form_submit_button("Add Entry"):
                entry_data = {
                    "product_name": product_name,
                    "category": category,
                    "quantity": quantity,
                    "unit_price": unit_price,
                    "supplier": supplier,
                    "purchase_date": purchase_date.isoformat(),
                    "expiry_date": expiry_date.isoformat(),
                    "batch_number": batch_number,
                    "notes": notes,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat()
                }
                
                if db.save_inventory_entry(entry_data):
                    st.success(f"Inventory entry for '{product_name}' added successfully!")
                    st.rerun()
                else:
                    st.error("Failed to add inventory entry.")
    
    with inv_tab3:
        st.subheader("Bulk Upload Inventory")
        
        # Download template button
        template = pd.DataFrame(columns=[
            'product_name', 'category', 'quantity', 'unit_price',
            'supplier', 'purchase_date', 'expiry_date', 'batch_number', 'notes'
        ])
        
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            template.to_excel(writer, index=False)
        
        st.download_button(
            label="📥 Download Template",
            data=buffer.getvalue(),
            file_name="inventory_template.xlsx",
            mime="application/vnd.ms-excel"
        )
        
        # Upload file
        uploaded_file = st.file_uploader("Choose a file", type=['xlsx'])
        if uploaded_file is not None:
            try:
                df = pd.read_excel(uploaded_file)
                st.write("Preview of uploaded data:")
                st.dataframe(df)
                
                if st.button("Process Upload"):
                    success_count = 0
                    for _, row in df.iterrows():
                        entry_data = row.to_dict()
                        entry_data['created_at'] = datetime.now().isoformat()
                        entry_data['updated_at'] = datetime.now().isoformat()
                        
                        if db.save_inventory_entry(entry_data):
                            success_count += 1
                    
                    st.success(f"Successfully processed {success_count} entries out of {len(df)}")
                    if success_count < len(df):
                        st.warning("Some entries failed to process. Please check the data and try again.")
                    st.rerun()
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

elif page == "Products":
    st.title("Product Management")
    
    # Create tabs for different product operations
    prod_tab1, prod_tab2 = st.tabs(["View/Edit Products", "Add New Product"])
    
    with prod_tab1:
        # Get and display products
        products = db.get_products()
        if products:
            for product in products:
                with st.expander(f"{product['name']} (Category: {product['category']})"):
                    col1, col2 = st.columns(2)
                    with col1:
                        edited_name = st.text_input("Product Name", product['name'], key=f"name_{product['_id']}")
                        edited_category = st.text_input("Category", product['category'], key=f"cat_{product['_id']}")
                    with col2:
                        edited_price = st.number_input("Base Price", product['base_price'], key=f"price_{product['_id']}")
                        edited_stock = st.number_input("Current Stock", product['initial_stock'], key=f"stock_{product['_id']}")
                    
                    if st.button("Update", key=f"update_{product['_id']}"):
                        if db.update_product(product['_id'], {
                            "name": edited_name,
                            "category": edited_category,
                            "base_price": edited_price,
                            "initial_stock": edited_stock,
                            "updated_at": datetime.now().isoformat()
                        }):
                            st.success("Product updated successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to update product.")
                    
                    if st.button("Delete", key=f"delete_{product['_id']}"):
                        if db.delete_product(product['_id']):
                            st.success("Product deleted successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to delete product.")
        else:
            st.info("No products found. Add some products to get started!")
    
    with prod_tab2:
        # Form to add new product
        with st.form("add_product_form"):
            new_name = st.text_input("Product Name")
            new_category = st.text_input("Category")
            new_price = st.number_input("Base Price", min_value=0.0)
            new_initial_stock = st.number_input("Initial Stock", min_value=0)
            
            if st.form_submit_button("Add Product"):
                if new_name and new_category:
                    product_data = {
                        "name": new_name,
                        "category": new_category,
                        "base_price": new_price,
                        "initial_stock": new_initial_stock,
                        "created_at": datetime.now().isoformat(),
                        "updated_at": datetime.now().isoformat()
                    }
                    if db.save_product(product_data):
                        st.success(f"Product '{new_name}' added successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to add product.")
                else:
                    st.error("Please fill in all required fields.")

elif page == "Categories":
    st.title("Category Management")
    
    # Create tabs for different category operations
    cat_tab1, cat_tab2 = st.tabs(["View/Edit Categories", "Add New Category"])
    
    with cat_tab1:
        # Get and display categories
        categories = db.get_categories()
        if categories:
            for category in categories:
                with st.expander(f"{category['name']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        edited_name = st.text_input("Category Name", category['name'], key=f"name_{category['_id']}")
                        edited_desc = st.text_area("Description", category.get('description', ''), key=f"desc_{category['_id']}")
                    with col2:
                        edited_margin = st.number_input("Target Margin (%)", category.get('target_margin', 30.0), key=f"margin_{category['_id']}")
                    
                    if st.button("Update", key=f"update_{category['_id']}"):
                        if db.update_category(category['_id'], {
                            "name": edited_name,
                            "description": edited_desc,
                            "target_margin": edited_margin,
                            "updated_at": datetime.now().isoformat()
                        }):
                            st.success("Category updated successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to update category.")
                    
                    if st.button("Delete", key=f"delete_{category['_id']}"):
                        if db.delete_category(category['_id']):
                            st.success("Category deleted successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to delete category.")
        else:
            st.info("No categories found. Add some categories to get started!")
    
    with cat_tab2:
        # Form to add new category
        with st.form("add_category_form"):
            new_name = st.text_input("Category Name")
            new_desc = st.text_area("Description")
            new_margin = st.number_input("Target Margin (%)", value=30.0)
            
            if st.form_submit_button("Add Category"):
                if new_name:
                    if db.save_category({
                        "name": new_name,
                        "description": new_desc,
                        "target_margin": new_margin,
                        "is_active": True,
                        "created_at": datetime.now().isoformat(),
                        "updated_at": datetime.now().isoformat()
                    }):
                        st.success(f"Category '{new_name}' added successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to add category.")
                else:
                    st.error("Please enter a category name.") 