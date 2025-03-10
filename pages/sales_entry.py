import streamlit as st
from datetime import datetime, timedelta
from database import MongoDB
from models.inventory_manager import InventoryManager
import random
import logging

logger = logging.getLogger(__name__)

def render_sales_entry():
    st.title("Sales Entry")
    
    # Initialize database and inventory manager
    db = MongoDB()
    inventory_manager = InventoryManager(db)
    
    def generate_inventory_data():
        """Helper function to generate sample inventory data with sufficient history for LSTM"""
        try:
            with st.spinner("Generating comprehensive inventory data..."):
                # Generate initial categories and products
                db.generate_sample_data(num_products=10, num_sales=0)  # First create products without sales
                
                # Generate historical sales data over the past 90 days
                end_date = datetime.utcnow()
                start_date = end_date - timedelta(days=90)
                
                # Generate 300 sales spread across 90 days (avg ~3.3 sales per day)
                dates = [start_date + timedelta(
                    days=random.randint(0, 90),
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59)
                ) for _ in range(300)]
                dates.sort()  # Sort dates chronologically
                
                # Generate sales for each date
                for sale_date in dates:
                    try:
                        sale_data = {
                            'customer': {
                                'name': f'Customer_{random.randint(1, 100)}',
                                'phone': f'{random.randint(1000000000, 9999999999)}',
                                'email': f'customer{random.randint(1, 100)}@example.com'
                            },
                            'items': [],
                            'payment_method': random.choice(['Cash', 'Card', 'UPI']),
                            'timestamp': sale_date,
                            'created_at': sale_date,
                            'updated_at': sale_date
                        }
                        
                        # Add 1-3 items per sale
                        num_items = random.randint(1, 3)
                        total_amount = 0
                        
                        # Get current product list
                        products = inventory_manager.get_all_products()
                        if not products:
                            raise ValueError("No products available")
                        
                        for _ in range(num_items):
                            product = random.choice(products)
                            quantity = random.randint(1, 5)
                            price = float(product['price'])
                            amount = quantity * price
                            
                            sale_data['items'].append({
                                'product_id': str(product['_id']),
                                'name': product['name'],
                                'quantity': quantity,
                                'price': price,
                                'amount': amount
                            })
                            total_amount += amount
                        
                        sale_data['total_amount'] = total_amount
                        
                        # Save the sale and update inventory
                        sale_id = inventory_manager.record_transaction(sale_data)
                        
                        # Update product quantities
                        for item in sale_data['items']:
                            inventory_manager.update_product(
                                item['product_id'],
                                {'quantity': -item['quantity']}
                            )
                        
                        # Randomly restock products (20% chance per product per day)
                        if random.random() < 0.2:
                            for product in products:
                                if product['quantity'] < product.get('min_stock', 50):
                                    restock_qty = random.randint(50, 200)
                                    inventory_manager.update_product(
                                        str(product['_id']),
                                        {'quantity': restock_qty}
                                    )
                    
                    except Exception as e:
                        logger.error(f"Error generating historical sale for date {sale_date}: {str(e)}")
                        continue
                
                return True
                
        except Exception as e:
            st.error(f"Error generating inventory data: {str(e)}")
            return False
    
    # Get product list and check if we need to generate sample data
    all_products = inventory_manager.get_all_products()
    
    # If no products exist, automatically generate sample data
    if not all_products:
        st.info("Initializing inventory with comprehensive historical data...")
        if generate_inventory_data():
            st.success("Initial inventory data generated successfully!")
            all_products = inventory_manager.get_all_products()
        else:
            st.error("Failed to initialize inventory. Please try again later.")
            return
    
    # Filter out out-of-stock items
    products = [p for p in all_products if p.get('quantity', 0) > 0]
    
    # If all products are out of stock, automatically generate more
    if not products:
        st.info("Replenishing inventory with new products...")
        if generate_inventory_data():
            st.success("Inventory replenished successfully!")
            # Refresh product list
            all_products = inventory_manager.get_all_products()
            products = [p for p in all_products if p.get('quantity', 0) > 0]
        else:
            st.error("Failed to replenish inventory. Please try again later.")
            return
    
    # Create form for sales entry
    with st.form("sales_entry_form", clear_on_submit=True):
        # Customer Information
        st.subheader("Customer Information")
        customer_name = st.text_input("Customer Name")
        customer_phone = st.text_input("Phone Number")
        customer_email = st.text_input("Email (optional)")
        
        # Sales Items
        st.subheader("Sales Items")
        num_items = st.number_input("Number of Items", min_value=1, max_value=min(10, len(products)), value=1)
        
        # Create dynamic lists for items
        items = []
        total_amount = 0
        selected_products = set()  # Track selected products to prevent duplicates
        
        for i in range(int(num_items)):
            st.write(f"Item {i + 1}")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Filter out already selected products
                available_products = [p for p in products if str(p['_id']) not in selected_products]
                if not available_products:
                    st.error("No more products available for selection")
                    break
                
                product = st.selectbox(
                    f"Product {i + 1}",
                    options=available_products,
                    format_func=lambda x: f"{x['name']} (Stock: {x['quantity']})",
                    key=f"product_{i}"
                )
            
            if product:
                with col2:
                    max_qty = int(product['quantity'])
                    quantity = st.number_input(
                        "Quantity",
                        min_value=1,
                        max_value=max_qty,
                        value=min(1, max_qty),
                        key=f"quantity_{i}",
                        help=f"Available stock: {max_qty}"
                    )
                
                with col3:
                    price = product['price']
                    amount = price * quantity
                    st.write(f"Amount: ₹{amount:.2f}")
                    total_amount += amount
                
                if quantity > 0:
                    items.append({
                        'product_id': str(product['_id']),
                        'name': product['name'],
                        'quantity': quantity,
                        'price': price,
                        'amount': amount,
                        'available_stock': max_qty  # Store available stock for validation
                    })
                    selected_products.add(str(product['_id']))  # Track selected product
        
        st.subheader(f"Total Amount: ₹{total_amount:.2f}")
        
        # Payment Information
        st.subheader("Payment Information")
        payment_method = st.selectbox(
            "Payment Method",
            options=["Cash", "Card", "UPI", "Other"]
        )
        
        # Submit button
        submitted = st.form_submit_button("Complete Sale", type="primary")
        
        if submitted:
            if not customer_name:
                st.error("Please enter customer name")
                return
            
            if not customer_phone:
                st.error("Please enter customer phone number")
                return
            
            if not items:
                st.error("Please add at least one item with quantity greater than 0")
                return
            
            # Validate stock levels again before processing
            try:
                with st.spinner("Validating stock levels..."):
                    stock_errors = []
                    for item in items:
                        current_stock = inventory_manager.get_product(item['product_id'])
                        if not current_stock:
                            stock_errors.append(f"Product {item['name']} no longer exists")
                            continue
                        if current_stock['quantity'] < item['quantity']:
                            stock_errors.append(
                                f"Insufficient stock for {item['name']}. "
                                f"Available: {current_stock['quantity']}, Requested: {item['quantity']}"
                            )
                    
                    if stock_errors:
                        st.error("Stock validation failed:")
                        for error in stock_errors:
                            st.warning(error)
                        # Automatically replenish inventory if stock is insufficient
                        st.info("Attempting to replenish inventory...")
                        if generate_inventory_data():
                            st.success("Inventory replenished. Please try your purchase again.")
                            st.experimental_rerun()
                        return
                
                # Create sale record
                sale_data = {
                    'customer': {
                        'name': customer_name,
                        'phone': customer_phone,
                        'email': customer_email
                    },
                    'items': [{k: v for k, v in item.items() if k != 'available_stock'} for item in items],
                    'total_amount': total_amount,
                    'payment_method': payment_method,
                    'timestamp': datetime.utcnow()
                }
                
                try:
                    with st.spinner("Processing sale..."):
                        # Record transaction first
                        sale_id = inventory_manager.record_transaction(sale_data)
                        
                        # Then update inventory for each item
                        for item in items:
                            try:
                                inventory_manager.update_product(
                                    item['product_id'],
                                    {'quantity': -item['quantity']}  # Decrease by quantity sold
                                )
                            except Exception as e:
                                st.error(f"Error updating inventory for {item['name']}: {str(e)}")
                                # Consider rolling back the transaction here
                                return
                    
                    st.success("Sale completed successfully!")
                    
                    # Generate receipt
                    st.subheader("Receipt")
                    receipt = f"""
                    **Sale ID:** {sale_id}
                    **Date:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}
                    
                    **Customer Information:**
                    Name: {customer_name}
                    Phone: {customer_phone}
                    {'Email: ' + customer_email if customer_email else ''}
                    
                    **Items:**
                    """
                    
                    for item in items:
                        receipt += f"\n{item['name']} x {item['quantity']} = ₹{item['amount']:.2f}"
                    
                    receipt += f"""
                    
                    **Total Amount:** ₹{total_amount:.2f}
                    **Payment Method:** {payment_method}
                    
                    Thank you for your purchase!
                    """
                    
                    st.markdown(receipt)
                    
                except Exception as e:
                    st.error(f"Error processing sale: {str(e)}")
                    return
                
            except Exception as e:
                st.error(f"Error validating stock: {str(e)}")
                return

if __name__ == "__main__":
    render_sales_entry() 