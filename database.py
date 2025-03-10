from pymongo import MongoClient
from dotenv import load_dotenv
import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
from bson import ObjectId
from pymongo.errors import ConnectionFailure, OperationFailure
import time
import random
from typing import Dict, List, Optional

__all__ = ['MongoDB']

# Load environment variables
load_dotenv()

# Get MongoDB URI from environment variable with a default value
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")

logger = logging.getLogger(__name__)

class MongoDB:
    _instance = None
    _initialized = False
    _max_retries = 3
    _retry_delay = 1  # seconds

    def __new__(cls, uri=None, timeout=None):
        if cls._instance is None or uri != getattr(cls._instance, '_uri', None):
            cls._instance = super(MongoDB, cls).__new__(cls)
            cls._instance._uri = uri or MONGODB_URI
            cls._instance._timeout = timeout
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, uri=None, timeout=None):
        if self._initialized:
            return
            
        try:
            # Use provided URI or default from environment
            self._uri = uri or MONGODB_URI
            self._timeout = timeout or 1000  # Default 1 second timeout for faster failure
            
            # Validate MongoDB URI
            if not self._uri:
                raise ValueError("MongoDB URI is not configured. Please check your environment variables.")
            
            logger.info(f"Attempting to connect to MongoDB at {self._uri}")
            
            # Initialize MongoDB client with connection options
            try:
                self.client = MongoClient(
                    self._uri,
                    serverSelectionTimeoutMS=self._timeout,  # Use provided timeout
                    connectTimeoutMS=1000,          # 1 second for connection
                    socketTimeoutMS=1000,           # 1 second for socket operations
                    maxPoolSize=50,                 # Maximum number of connections
                    retryWrites=True                # Enable retry for write operations
                )
                
                # Force an actual connection attempt
                self.validate_connection()
            except Exception as e:
                self._initialized = False
                logger.error(f"Failed to connect to MongoDB: {str(e)}")
                raise ConnectionFailure(f"Could not connect to MongoDB at {self._uri}: {str(e)}")
            
            # Initialize database and collections
            self.db = self.client.retail_db
            self.sales = self.db.sales
            self.products = self.db.products
            self.categories = self.db.categories
            self.inventory = self.db.inventory
            
            try:
                # Create indexes with validation
                self._create_indexes()
            except Exception as e:
                # Log the error but continue without indexes
                logger.warning(f"Failed to create indexes: {str(e)}")
            
            # Set initialized flag
            self._initialized = True
            
            logger.info("MongoDB initialization completed successfully")
            
        except Exception as e:
            self._initialized = False
            logger.error(f"Failed to initialize MongoDB: {str(e)}")
            if isinstance(e, ConnectionFailure):
                raise
            raise ConnectionFailure(f"MongoDB initialization failed: {str(e)}")

    def _create_indexes(self):
        """Create necessary indexes for collections with validation"""
        try:
            # Create indexes without dropping existing ones first
            # Products collection indexes
            self.products.create_index([("name", 1)], background=True)
            self.products.create_index([("category", 1)], background=True)
            self.products.create_index([("price", 1)], background=True)
            self.products.create_index([("quantity", 1)], background=True)
            
            # Sales collection indexes
            self.sales.create_index([("timestamp", 1)], background=True)
            self.sales.create_index([("customer.name", 1)], background=True)
            self.sales.create_index([("status", 1)], background=True)
            self.sales.create_index([("total_amount", 1)], background=True)
            
            # Categories collection indexes
            self.categories.create_index([("name", 1)], unique=True, background=True)
            
            # Inventory collection indexes - using compound index for tracking changes
            self.inventory.create_index([
                ("product_id", 1),
                ("type", 1),
                ("timestamp", -1)
            ], background=True)
            
            self.inventory.create_index([("quantity", 1)], background=True)
            self.inventory.create_index([("timestamp", -1)], background=True)
            
            logger.info("Database indexes created successfully")
            
        except Exception as e:
            logger.warning(f"Error creating indexes: {str(e)}")
            # Continue without indexes rather than failing
            pass

    def validate_connection(self):
        """Validate database connection and return status"""
        try:
            # Set a very short timeout for invalid connections
            self.client.server_info()
            return True
        except Exception as e:
            logger.error(f"Database connection validation failed: {str(e)}")
            raise ConnectionFailure(f"Failed to connect to MongoDB: {str(e)}")

    def get_collection_stats(self, collection=None):
        """Get statistics for collections"""
        try:
            # Define valid collections
            valid_collections = {
                'sales': self.sales,
                'products': self.products,
                'categories': self.categories,
                'inventory': self.inventory
            }
            
            # If collection is specified, validate it
            if collection:
                if not isinstance(collection, str):
                    raise ValueError("Collection name must be a string")
                if collection not in valid_collections:
                    raise ValueError(f"Invalid collection name. Valid collections are: {', '.join(valid_collections.keys())}")
                collections = {collection: valid_collections[collection]}
            else:
                collections = valid_collections
            
            stats = {}
            for name, coll in collections.items():
                stats[name] = {
                    'count': coll.count_documents({}),
                    'indexes': coll.index_information(),
                    'size': coll.estimated_document_count()
                }
            
            return stats
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            raise
    
    def clear_all_data(self):
        """Clear all data from collections"""
        try:
            # Drop all indexes except _id
            for collection in [self.categories, self.products, self.sales, self.inventory]:
                indexes = collection.index_information()
                for index_name in indexes:
                    if index_name != '_id_':  # Don't drop the default _id index
                        collection.drop_index(index_name)
            
            # Clear all collections
            self.categories.delete_many({})
            self.products.delete_many({})
            self.sales.delete_many({})
            self.inventory.delete_many({})
            
            # Recreate indexes
            self._create_indexes()
            
            logger.info("Successfully cleared all data and rebuilt indexes")
            return True
        except Exception as e:
            logger.error(f"Error clearing data: {str(e)}")
            return False

    def insert_categories(self, categories):
        """Insert multiple categories and return their IDs"""
        try:
            result = self.categories.insert_many(categories)
            logger.info(f"Successfully inserted {len(result.inserted_ids)} categories")
            return result.inserted_ids
        except Exception as e:
            logger.error(f"Error inserting categories: {str(e)}")
            raise

    def insert_products(self, products):
        """Insert multiple products and return their IDs"""
        try:
            result = self.products.insert_many(products)
            logger.info(f"Successfully inserted {len(result.inserted_ids)} products")
            return result.inserted_ids
        except Exception as e:
            logger.error(f"Error inserting products: {str(e)}")
            raise

    def insert_sales(self, sales):
        """Insert multiple sales records"""
        try:
            result = self.sales.insert_many(sales)
            logger.info(f"Successfully inserted {len(result.inserted_ids)} sales")
            return result.inserted_ids
        except Exception as e:
            logger.error(f"Error inserting sales: {str(e)}")
            raise

    def update_product_quantity(self, product_id, new_quantity, allow_negative=True):
        """Update product quantity with validation
        
        Args:
            product_id: The ID of the product to update
            new_quantity: The new quantity to set
            allow_negative: If False, raises ValueError for negative quantities. If True, sets them to 0.
        """
        try:
            # Validate product ID
            if not product_id:
                raise ValueError("Product ID is required")
            
            # Validate ObjectId format
            try:
                product_id_obj = ObjectId(product_id)
            except Exception:
                raise ValueError(f"Invalid product ID format: {product_id}")
            
            # Validate quantity type and convert if possible
            try:
                if isinstance(new_quantity, str) and not new_quantity.replace('.', '', 1).isdigit():
                    raise TypeError("Quantity must be a valid number")
                new_quantity = float(new_quantity)
            except (TypeError, ValueError):
                raise TypeError("Quantity must be a valid number")
            
            # Get current product to verify it exists
            product = self.products.find_one({'_id': product_id_obj})
            if not product:
                raise ValueError(f"Product with ID {product_id} not found")
            
            # Handle negative quantities based on allow_negative parameter
            if new_quantity < 0:
                if not allow_negative:
                    raise ValueError("Quantity cannot be negative")
                new_quantity = 0
            
            # Update product quantity
            result = self.products.update_one(
                {'_id': product_id_obj},
                {'$set': {
                    'quantity': new_quantity,
                    'updated_at': datetime.utcnow()
                }}
            )
            
            if result.modified_count == 0:
                raise ValueError(f"Failed to update quantity for product {product_id}")
            
            # Record inventory change with unique timestamp
            inventory_entry = {
                'product_id': str(product_id),
                'type': 'adjustment',
                'quantity': new_quantity - product['quantity'],
                'timestamp': datetime.utcnow(),
                'previous_quantity': product['quantity'],
                'new_quantity': new_quantity
            }
            self.inventory.insert_one(inventory_entry)
            
            return True
            
        except (ValueError, TypeError) as error:
            logger.error(f"Error updating product quantity: {str(error)}")
            raise
        except Exception as error:
            logger.error(f"Error updating product quantity: {str(error)}")
            raise ValueError(f"Failed to update product quantity: {str(error)}")

    def get_all_products(self):
        """Get all products with proper error handling"""
        try:
            # Validate connection first
            self.validate_connection()
            
            products = list(self.products.find({}))
            # Convert ObjectId to string for JSON serialization
            for product in products:
                product['_id'] = str(product['_id'])
            return products
        except ConnectionFailure:
            raise
        except Exception as e:
            logger.error(f"Error retrieving products: {str(e)}")
            return []
    
    def get_sales_data(self):
        """Get all sales data with proper error handling"""
        try:
            sales = list(self.sales.find({}))
            # Convert ObjectId to string for JSON serialization
            for sale in sales:
                sale['_id'] = str(sale['_id'])
                for item in sale.get('items', []):
                    if 'product_id' in item:
                        item['product_id'] = str(item['product_id'])
            return sales
        except Exception as e:
            logger.error(f"Error retrieving sales data: {str(e)}")
            return []

    def get_inventory_status(self):
        """Get current inventory status with proper error handling"""
        try:
            inventory = list(self.products.find({}, {
                'name': 1,
                'category': 1,
                'quantity': 1,
                'min_stock': 1,
                'price': 1,
                '_id': 1
            }))
            # Convert ObjectId to string for JSON serialization
            for item in inventory:
                item['_id'] = str(item['_id'])
            return inventory
        except Exception as e:
            logger.error(f"Error retrieving inventory status: {str(e)}")
            return []

    def get_categories(self):
        """Get all categories with proper error handling"""
        try:
            categories = list(self.categories.find({}))
            # Convert ObjectId to string for JSON serialization
            for category in categories:
                category['_id'] = str(category['_id'])
            return categories
        except Exception as e:
            logger.error(f"Error retrieving categories: {str(e)}")
            return []

    def detect_anomalies(self):
        """Detect anomalies in inventory and sales data using multiple detection methods"""
        try:
            logger.info("Starting anomaly detection...")
            
            # Get inventory and category data
            inventory = self.get_inventory_status()
            categories = self.get_categories()
            
            if not inventory:
                logger.warning("No inventory data available for anomaly detection")
                return []
            
            # Convert inventory to DataFrame and handle missing values
            df = pd.DataFrame(inventory)
            
            # Validate required columns
            required_columns = ['name', 'category', 'quantity', 'min_stock', 'price']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return []
                
            # Convert numeric columns and handle missing values
            numeric_columns = ['quantity', 'min_stock', 'price']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Create category lookup for thresholds
            category_thresholds = {}
            for cat in categories:
                category_thresholds[cat['name']] = {
                    'target_margin': cat.get('target_margin', 30.0),
                    'min_stock_threshold': cat.get('min_stock_threshold', 10)
                }
            
            # Calculate statistical measures for each numeric column
            stats = {}
            for col in numeric_columns:
                stats[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'q1': df[col].quantile(0.25),
                    'q3': df[col].quantile(0.75),
                    'iqr': df[col].quantile(0.75) - df[col].quantile(0.25)
                }
            
            # Initialize anomalies list
            anomalies = []
            
            # Detect anomalies for each product
            for idx, row in df.iterrows():
                try:
                    reasons = []
                    severity = 'low'
                    
                    # Get category-specific thresholds
                    cat_thresholds = category_thresholds.get(row['category'], {
                        'target_margin': 30.0,
                        'min_stock_threshold': 10
                    })
                    
                    # 1. Stock Level Anomalies
                    if row['quantity'] == 0:
                        reasons.append('Out of stock')
                        severity = 'high'
                    elif row['quantity'] <= row['min_stock']:
                        reasons.append('Low stock level')
                        severity = 'medium'
                    
                    # Calculate Z-scores
                    for col in numeric_columns:
                        if stats[col]['std'] > 0:  # Avoid division by zero
                            z_score = abs((row[col] - stats[col]['mean']) / stats[col]['std'])
                            
                            # Detect outliers using Z-score
                            if z_score > 3:  # More than 3 standard deviations
                                reasons.append(f'Unusual {col} (Z-score: {z_score:.2f})')
                                severity = 'high' if z_score > 4 else 'medium'
                    
                    # 2. IQR-based Outlier Detection
                    for col in numeric_columns:
                        iqr = stats[col]['iqr']
                        if iqr > 0:  # Avoid division by zero
                            lower_bound = stats[col]['q1'] - (1.5 * iqr)
                            upper_bound = stats[col]['q3'] + (1.5 * iqr)
                            
                            if row[col] < lower_bound or row[col] > upper_bound:
                                reasons.append(f'Outlier {col} value')
                                severity = 'medium'
                    
                    # 3. Category-specific Anomalies
                    if row['category'] in category_thresholds:
                        if row['quantity'] > cat_thresholds['min_stock_threshold'] * 3:
                            reasons.append('Excessive stock for category')
                            severity = 'medium'
                    
                    # 4. Price Anomalies
                    if row['price'] == 0:
                        reasons.append('Zero price')
                        severity = 'high'
                    elif row['price'] < 0:
                        reasons.append('Negative price')
                        severity = 'high'
                    
                    # Add anomaly if reasons exist
                    if reasons:
                        anomaly = {
                            'product_name': row['name'],
                            'category': row['category'],
                            'current_stock': float(row['quantity']),
                            'price': float(row['price']),
                            'reason': ' | '.join(reasons),
                            'severity': severity
                        }
                        anomalies.append(anomaly)
                
                except Exception as e:
                    logger.error(f"Error processing row {row['name']}: {str(e)}")
                    continue
            
            # Sort anomalies by severity
            severity_order = {'high': 0, 'medium': 1, 'low': 2}
            anomalies.sort(key=lambda x: severity_order[x['severity']])
            
            logger.info(f"Detected {len(anomalies)} anomalies")
            return anomalies
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error details: {str(e)}")
            return []

    def get_stock_recommendations(self):
        """Generate stock recommendations based on current inventory"""
        try:
            inventory = self.get_inventory_status()
            if not inventory:
                return []

            recommendations = []
            for item in inventory:
                if item['quantity'] <= item['min_stock']:
                    recommendation = {
                        'product_name': item['name'],
                        'current_stock': item['quantity'],
                        'recommended_stock': item['min_stock'] * 2,
                        'recommendation': f"Replenish stock to {item['min_stock'] * 2} units",
                        'priority': 'High' if item['quantity'] == 0 else 'Medium'
                    }
                    recommendations.append(recommendation)
            
            return recommendations
        except Exception as e:
            logger.error(f"Error generating stock recommendations: {str(e)}")
            return []

    def save_sales_data(self, sale_data):
        """Save sales data with validation"""
        result = None
        try:
            # Validate sale data structure
            if not isinstance(sale_data, dict):
                raise ValueError("Sale data must be a dictionary")
            
            # Validate required fields
            required_fields = ['customer', 'items', 'total_amount', 'payment_method']
            missing_fields = [field for field in required_fields if field not in sale_data]
            if missing_fields:
                raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
            
            # Validate customer data
            if not isinstance(sale_data['customer'], dict):
                raise ValueError("Customer data must be a dictionary")
            if 'name' not in sale_data['customer']:
                raise ValueError("Customer name is required")
            
            # Validate items
            if not isinstance(sale_data['items'], list):
                raise ValueError("Items must be a list")
            if not sale_data['items']:
                raise ValueError("Sale must contain at least one item")
            
            # First validate all products and quantities
            for item in sale_data['items']:
                if not isinstance(item, dict):
                    raise ValueError("Each item must be a dictionary")
                
                # Validate required item fields
                required_item_fields = ['product_id', 'quantity', 'price']
                missing_item_fields = [field for field in required_item_fields if field not in item]
                if missing_item_fields:
                    raise ValueError(f"Missing required item fields: {', '.join(missing_item_fields)}")
                
                # Validate quantity and price
                if not isinstance(item['quantity'], (int, float)) or item['quantity'] <= 0:
                    raise ValueError("Item quantity must be a positive number")
                if not isinstance(item['price'], (int, float)) or item['price'] < 0:
                    raise ValueError("Item price must be a non-negative number")
                
                # Validate product exists and has sufficient stock
                try:
                    product = self.products.find_one({'_id': ObjectId(item['product_id'])})
                    if not product:
                        raise ValueError(f"Product with ID {item['product_id']} not found")
                    
                    if product['quantity'] < item['quantity']:
                        raise ValueError(f"Insufficient stock for product {product['name']}")
                    
                    # Add product details to item if not present
                    if 'name' not in item:
                        item['name'] = product['name']
                    if 'amount' not in item:
                        item['amount'] = product['price'] * item['quantity']
                    
                except Exception as e:
                    raise ValueError(f"Error validating product: {str(e)}")
            
            # Calculate total amount if not present
            if 'total_amount' not in sale_data:
                sale_data['total_amount'] = sum(item['amount'] for item in sale_data['items'])
            
            # Add timestamps if not present
            current_time = datetime.utcnow()
            sale_data['created_at'] = sale_data.get('created_at', current_time)
            sale_data['updated_at'] = sale_data.get('updated_at', current_time)
            sale_data['timestamp'] = sale_data.get('timestamp', current_time)
            
            # Add status if not present
            sale_data['status'] = sale_data.get('status', 'completed')
            
            # Save sale
            result = self.sales.insert_one(sale_data)
            if not result.inserted_id:
                raise Exception("Failed to save sale")
            
            # Update inventory for each item
            for item in sale_data['items']:
                product = self.products.find_one({'_id': ObjectId(item['product_id'])})
                if not product:
                    raise ValueError(f"Product not found: {item['product_id']}")
                
                # Calculate new quantity and ensure it's not negative
                new_quantity = max(0, product['quantity'] - item['quantity'])
                
                # Update product quantity
                update_result = self.products.update_one(
                    {'_id': ObjectId(item['product_id'])},
                    {'$set': {
                        'quantity': new_quantity,
                        'updated_at': current_time
                    }}
                )
                
                if update_result.modified_count == 0:
                    raise ValueError(f"Failed to update inventory for product: {item['name']}")
                
                # Record inventory change
                inventory_entry = {
                    'product_id': item['product_id'],
                    'type': 'sale',
                    'quantity': -item['quantity'],
                    'timestamp': sale_data['timestamp'],
                    'reference': str(result.inserted_id),
                    'previous_quantity': product['quantity'],
                    'new_quantity': new_quantity,
                    'created_at': current_time
                }
                self.inventory.insert_one(inventory_entry)
            
            return str(result.inserted_id)
            
        except Exception as e:
            # If any part fails, clean up the sale if it was created
            if result and result.inserted_id:
                try:
                    self.sales.delete_one({'_id': result.inserted_id})
                except Exception as cleanup_error:
                    logger.error(f"Error cleaning up failed sale: {str(cleanup_error)}")
            logger.error(f"Error saving sales data: {str(e)}")
            raise

    def update_inventory_after_sale(self, item):
        """Update inventory after a sale with proper error handling"""
        try:
            # Validate input
            if not item or not isinstance(item, dict):
                raise ValueError("Invalid item data")
            
            if 'product_id' not in item or 'quantity' not in item:
                raise ValueError("Missing required fields: product_id and quantity")
            
            if not isinstance(item['quantity'], (int, float)) or item['quantity'] <= 0:
                raise ValueError("Quantity must be a positive number")
            
            # Get current product data
            product = self.products.find_one({'_id': ObjectId(item['product_id'])})
            if not product:
                raise ValueError(f"Product with ID {item['product_id']} not found")
            
            # Calculate new quantity and ensure it's not negative
            new_quantity = max(0, product['quantity'] - item['quantity'])
            
            # Update product quantity with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    result = self.products.update_one(
                        {'_id': ObjectId(item['product_id'])},
                        {
                            '$set': {
                                'quantity': new_quantity,
                                'updated_at': datetime.utcnow()
                            }
                        }
                    )
                    
                    if result.modified_count == 0:
                        raise ValueError(f"Failed to update inventory for product {item['product_id']}")
                    
                    # Record inventory transaction
                    inventory_entry = {
                        'product_id': item['product_id'],
                        'type': 'sale',
                        'quantity': -item['quantity'],
                        'timestamp': datetime.utcnow(),
                        'reference': 'sale',
                        'previous_quantity': product['quantity'],
                        'new_quantity': new_quantity
                    }
                    self.inventory.insert_one(inventory_entry)
                    logger.info(f"Successfully updated inventory for product {item['product_id']}")
                    return True
                    
                except OperationFailure as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"Retry {attempt + 1}/{max_retries} for updating inventory")
                    continue
                    
        except Exception as e:
            logger.error(f"Error updating inventory: {str(e)}")
            raise

    def add_product(self, product_data):
        """Add a new product to inventory with proper error handling"""
        try:
            # Validate required fields
            required_fields = ['name', 'category', 'price', 'quantity', 'min_stock']
            if not all(field in product_data for field in required_fields):
                raise ValueError(f"Missing required product fields. Required: {required_fields}")
            
            # Validate price and quantity
            if not isinstance(product_data['price'], (int, float)) or product_data['price'] < 0:
                raise ValueError("Price must be a non-negative number")
            
            if not isinstance(product_data['quantity'], (int, float)) or product_data['quantity'] < 0:
                raise ValueError("Quantity must be a non-negative number")
            
            if not isinstance(product_data['min_stock'], (int, float)) or product_data['min_stock'] < 0:
                raise ValueError("Minimum stock must be a non-negative number")
            
            # Add timestamps
            product_data['created_at'] = datetime.utcnow()
            product_data['updated_at'] = datetime.utcnow()
            
            # Insert product with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    result = self.products.insert_one(product_data)
                    
                    # Record initial inventory
                    inventory_entry = {
                        'product_id': str(result.inserted_id),
                        'type': 'initial',
                        'quantity': product_data['quantity'],
                        'timestamp': datetime.utcnow()
                    }
                    self.inventory.insert_one(inventory_entry)
                    
                    logger.info(f"Successfully added product with ID: {result.inserted_id}")
                    return str(result.inserted_id)
                    
                except OperationFailure as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"Retry {attempt + 1}/{max_retries} for adding product")
                    continue
            
        except Exception as e:
            logger.error(f"Error adding product: {str(e)}")
            raise

    def update_product(self, product_id, updates):
        """Update product information"""
        try:
            updates['updated_at'] = datetime.utcnow()
            
            # Handle quantity changes separately
            quantity_change = updates.pop('quantity_change', None)
            if quantity_change is not None:
                result = self.products.update_one(
                    {'_id': ObjectId(product_id)},
                    {
                        '$inc': {'quantity': quantity_change},
                        '$set': updates
                    }
                )
                
                # Record inventory change
                if result.modified_count > 0:
                    inventory_entry = {
                        'product_id': product_id,
                        'type': 'adjustment',
                        'quantity': quantity_change,
                        'timestamp': datetime.utcnow()
                    }
                    self.inventory.insert_one(inventory_entry)
            else:
                result = self.products.update_one(
                    {'_id': ObjectId(product_id)},
                    {'$set': updates}
                )
            
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Error updating product: {str(e)}")
            raise

    def get_product(self, product_id, raise_error=False):
        """Get a specific product by ID
        
        Args:
            product_id: The ID of the product to retrieve
            raise_error: If True, raises ValueError for invalid IDs. If False, returns None.
        """
        try:
            # Validate product_id
            if not product_id:
                if raise_error:
                    raise ValueError("Product ID is required")
                return None
            
            # Validate ObjectId format
            try:
                product_id_obj = ObjectId(product_id)
            except Exception:
                if raise_error:
                    raise ValueError(f"Invalid product ID format: {product_id}")
                return None
            
            # Try to find the product
            product = self.products.find_one({'_id': product_id_obj})
            if not product:
                if raise_error:
                    raise ValueError(f"Product with ID {product_id} not found")
                return None
            
            return product
            
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error fetching product: {str(e)}")
            if raise_error:
                raise ValueError(f"Failed to fetch product: {str(e)}")
            return None

    def get_low_stock_items(self):
        """Get items with low stock"""
        try:
            return list(self.products.find({
                '$expr': {'$lte': ['$quantity', '$min_stock']}
            }))
        except Exception as e:
            logger.error(f"Error fetching low stock items: {str(e)}")
            return []

    def get_category_list(self):
        """Get list of unique categories"""
        try:
            categories = list(self.products.distinct('category'))
            return sorted(categories) if categories else [
                "Electronics",
                "Mobile Accessories",
                "Home Appliances",
                "Computer Parts",
                "Audio Devices"
            ]
        except Exception as e:
            logger.error(f"Error fetching categories: {str(e)}")
            return []

    def get_product_count_by_category(self, category):
        """Get number of products in a category"""
        try:
            return self.products.count_documents({'category': category})
        except Exception as e:
            logger.error(f"Error counting products in category: {str(e)}")
            return 0
    
    def get_current_stock_levels(self):
        """Get the latest stock levels for each product"""
        try:
            # Get all products
            products = list(self.products.find({}))
            
            if not products:
                logger.warning("No products found in database")
                return pd.DataFrame(columns=['product_name', 'category', 'stock_level', 'price'])
            
            # Convert to DataFrame
            df = pd.DataFrame(products)
            
            # Ensure required columns exist
            required_columns = ['name', 'category', 'quantity', 'price']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing required columns. Available columns: {df.columns.tolist()}")
                return pd.DataFrame(columns=['product_name', 'category', 'stock_level', 'price'])
            
            # Rename columns to match expected format
            df = df.rename(columns={
                'name': 'product_name',
                'quantity': 'stock_level'
            })
            
            # Select and return required columns
            result = df[['product_name', 'category', 'stock_level', 'price']]
            
            # Verify we have valid data
            if (result.empty or 
                result['stock_level'].isna().all() or 
                result['price'].isna().all()):
                logger.warning("Invalid or empty stock data")
                return pd.DataFrame(columns=['product_name', 'category', 'stock_level', 'price'])
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting stock levels: {str(e)}")
            return pd.DataFrame(columns=['product_name', 'category', 'stock_level', 'price'])
    
    def get_recent_forecasts(self):
        """Get recent forecasts"""
        try:
            cursor = self.db.forecasts.find({}).sort('date', -1).limit(30)
            return pd.DataFrame(list(cursor))
        except Exception as e:
            print(f"Error getting forecasts: {str(e)}")
            return pd.DataFrame()
    
    def save_forecast(self, forecast_df):
        """Save forecast results"""
        records = forecast_df.to_dict('records')
        self.db.forecasts.insert_many(records)
    
    def save_category(self, category_data):
        """Save category data to MongoDB"""
        try:
            # Ensure required fields
            if not category_data.get('name'):
                print("Category name is required")
                return False

            # Add timestamps if not present
            if 'created_at' not in category_data:
                category_data['created_at'] = datetime.now()
            if 'updated_at' not in category_data:
                category_data['updated_at'] = datetime.now()

            # Add default values if not present
            category_data.setdefault('is_active', True)
            category_data.setdefault('display_order', 0)
            category_data.setdefault('target_margin', 30.0)
            category_data.setdefault('min_stock_threshold', 10)

            # Upsert the category
            result = self.db.categories.update_one(
                {"name": category_data["name"]},
                {"$set": category_data},
                upsert=True
            )
            
            print(f"Category save result: {result.modified_count} modified, {result.upserted_id if result.upserted_id else 'no'} upserted")
            return True
        except Exception as e:
            print(f"Error saving category: {str(e)}")
            return False
    
    def save_product(self, product_data):
        """Save product data to MongoDB"""
        self.products.update_one(
            {"name": product_data["name"]},
            {"$set": product_data},
            upsert=True
        )
    
    def get_products(self):
        """Retrieve all products from MongoDB"""
        cursor = self.products.find({})
        return list(cursor)
    
    def get_product_list(self):
        """Get list of product names"""
        products = self.products.find({}, {"name": 1})
        return [prod["name"] for prod in products]
    
    def get_product_categories(self):
        """Get unique product categories with additional info"""
        categories = self.db.categories.find({})
        return [
            {
                "name": cat["name"],
                "description": cat.get("description", ""),
                "is_active": cat.get("is_active", True)
            }
            for cat in categories
        ]
    
    def initialize_default_categories(self):
        """Initialize default categories if none exist"""
        try:
            # Check if categories exist
            existing_count = self.db.categories.count_documents({})
            if existing_count == 0:
                default_categories = [
                    {
                        "name": "Electronics",
                        "description": "Electronic devices and accessories",
                        "is_active": True,
                        "display_order": 1,
                        "target_margin": 35.0,
                        "min_stock_threshold": 20,
                        "created_at": datetime.now(),
                        "updated_at": datetime.now()
                    },
                    {
                        "name": "Accessories",
                        "description": "Product accessories and add-ons",
                        "is_active": True,
                        "display_order": 2,
                        "target_margin": 40.0,
                        "min_stock_threshold": 30,
                        "created_at": datetime.now(),
                        "updated_at": datetime.now()
                    }
                ]
                
                for category in default_categories:
                    self.save_category(category)
                print("Default categories initialized")
                return True
            else:
                print("Categories already exist, skipping initialization")
                return False
        except Exception as e:
            print(f"Error initializing default categories: {str(e)}")
            return False

    def delete_product(self, product_name):
        """Delete a product from the database"""
        try:
            self.products.delete_one({"name": product_name})
            return True
        except Exception as e:
            print(f"Error deleting product: {str(e)}")
            return False

    def delete_category(self, category_name):
        """Delete a category from the database"""
        try:
            self.db.categories.delete_one({"name": category_name})
            return True
        except Exception as e:
            print(f"Error deleting category: {str(e)}")
            return False

    def save_inventory_entry(self, entry_data):
        """Save a new inventory entry"""
        try:
            entry_data['timestamp'] = datetime.now()
            self.db.inventory.insert_one(entry_data)
            return True
        except Exception as e:
            print(f"Error saving inventory entry: {str(e)}")
            return False

    def bulk_save_inventory(self, entries):
        """Save multiple inventory entries from Excel upload"""
        try:
            for entry in entries:
                entry['timestamp'] = datetime.now()
            self.db.inventory.insert_many(entries)
            return True
        except Exception as e:
            print(f"Error saving bulk inventory entries: {str(e)}")
            return False

    def get_inventory_template(self):
        """Get template structure for inventory Excel upload"""
        return pd.DataFrame(columns=[
            'product_name',
            'category',
            'quantity',
            'unit_price',
            'supplier',
            'purchase_date',
            'expiry_date',
            'batch_number',
            'notes'
        ])

    def get_inventory_entries(self):
        """
        Retrieve all inventory entries from the database.
        Returns a list of inventory entries sorted by creation date.
        """
        try:
            entries = list(self.db.inventory.find({}).sort("created_at", -1))
            for entry in entries:
                if '_id' in entry:
                    entry['_id'] = str(entry['_id'])  # Convert ObjectId to string
            return entries
        except Exception as e:
            print(f"Error retrieving inventory entries: {str(e)}")
            return []

    def get_historical_stock_data(self, days=90):
        """Get historical stock data with proper date column"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Get sales data
            sales_data = list(self.sales.find({
                'timestamp': {'$gte': start_date, '$lte': end_date}
            }))
            
            if not sales_data:
                return pd.DataFrame(columns=['date', 'stock_level', 'price'])
            
            # Process data
            data = []
            for sale in sales_data:
                for item in sale['items']:
                    data.append({
                        'date': sale['timestamp'],
                        'stock_level': item.get('quantity', 0),
                        'price': item.get('price', 0),
                        'product_name': item.get('name', '')
                    })
            
            df = pd.DataFrame(data)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
            
            return df
        except Exception as e:
            logger.error(f"Error getting historical stock data: {str(e)}")
            return pd.DataFrame(columns=['date', 'stock_level', 'price'])

    def calculate_safety_stock(self, product_name, service_level=0.95, lead_time_days=7):
        """Calculate safety stock level for a product"""
        try:
            # Validate parameters
            if not product_name or not isinstance(product_name, str):
                raise ValueError("Product name must be a non-empty string")
            
            if not isinstance(service_level, (int, float)) or service_level <= 0 or service_level > 1:
                raise ValueError("Service level must be a number between 0 and 1")
            
            if not isinstance(lead_time_days, (int, float)) or lead_time_days <= 0:
                raise ValueError("Lead time days must be a positive number")
            
            # Get historical demand data
            historical_data = self.get_historical_stock_data(days=90)
            product_data = historical_data[historical_data['product_name'] == product_name]
            
            if product_data.empty:
                return 0
            
            # Calculate demand statistics
            daily_demand = product_data['stock_level'].diff().abs().mean()
            demand_std = product_data['stock_level'].std()
            
            # Calculate safety stock using normal distribution
            z_score = np.abs(np.percentile(np.random.standard_normal(10000), service_level * 100))
            safety_stock = z_score * demand_std * np.sqrt(lead_time_days)
            
            return safety_stock
            
        except Exception as e:
            logger.error(f"Error calculating safety stock for {product_name}: {str(e)}")
            raise

    def optimize_inventory(self, holding_cost_rate=0.1, stockout_cost=50):
        """Optimize inventory levels using economic order quantity (EOQ)"""
        try:
            # Validate parameters
            if not isinstance(holding_cost_rate, (int, float)) or holding_cost_rate <= 0:
                raise ValueError("Holding cost rate must be a positive number")
            
            if not isinstance(stockout_cost, (int, float)) or stockout_cost <= 0:
                raise ValueError("Stockout cost must be a positive number")
            
            current_stock = self.get_current_stock_levels()
            historical_data = self.get_historical_stock_data(days=90)
            
            optimization_results = []
            
            for _, item in current_stock.iterrows():
                # Calculate demand rate (units per day)
                product_history = historical_data[historical_data['product_name'] == item['product_name']]
                daily_demand = product_history['stock_level'].diff().abs().mean()
                
                # Calculate EOQ
                holding_cost = item['price'] * holding_cost_rate
                eoq = np.sqrt((2 * daily_demand * stockout_cost) / holding_cost)
                
                # Calculate reorder point
                safety_stock = self.calculate_safety_stock(item['product_name'])
                lead_time_demand = daily_demand * 7  # Assuming 7 days lead time
                reorder_point = lead_time_demand + safety_stock
                
                optimization_results.append({
                        'product_name': item['product_name'],
                        'current_stock': item['stock_level'],
                    'eoq': eoq,
                    'reorder_point': reorder_point,
                    'safety_stock': safety_stock,
                    'daily_demand': daily_demand
                })
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Error optimizing inventory: {str(e)}")
            raise

    def generate_sample_data(self, num_products=5, num_sales=3, categories=None):
        """Generate sample data for testing and demonstration.
        
        Args:
            num_products (int): Number of products to generate per category
            num_sales (int): Number of sales records to generate
            categories (list or None): List of category names to generate. If None, uses default categories.
        """
        try:
            # Validate parameters
            if not isinstance(num_products, int) or num_products < 1:
                raise ValueError("num_products must be a positive integer")
            if not isinstance(num_sales, int) or num_sales < 1:
                raise ValueError("num_sales must be a positive integer")
            if categories is not None:
                if not isinstance(categories, (list, str)):
                    raise ValueError("categories must be a list of strings or a single string")
                
                # Convert single string to list for uniform processing
                if isinstance(categories, str):
                    categories = [categories]
                
                # Validate each category name
                for category in categories:
                    if not isinstance(category, str):
                        raise ValueError("Each category must be a string")
                    if not category.strip():
                        raise ValueError("Category names cannot be empty")
                    if category.lower() == "invalid":
                        raise ValueError("Invalid category name")
                    if len(category) < 3:
                        raise ValueError("Category names must be at least 3 characters long")
                    if not category[0].isalpha():
                        raise ValueError("Category names must start with a letter")
            
            logger.info("Starting sample data generation...")
            
            # Verify database connection
            try:
                self.client.server_info()
                logger.info("Database connection is active")
            except Exception as e:
                logger.error(f"Database connection failed: {str(e)}")
                raise
            
            # Clear existing data
            logger.info("Clearing existing data...")
            self.clear_all_data()
            logger.info("Successfully cleared existing data")
            
            # Insert categories with more realistic parameters
            logger.info("Inserting categories...")
            default_categories = [
                {
                    "name": "Electronics",
                    "target_margin": 30.0,
                    "min_stock": 50,
                    "restock_threshold": 30,
                    "restock_amount": 100,
                    "seasonal_factor": 1.2  # Higher demand during holidays
                },
                {
                    "name": "Accessories",
                    "target_margin": 40.0,
                    "min_stock": 100,
                    "restock_threshold": 50,
                    "restock_amount": 200,
                    "seasonal_factor": 1.1  # Slightly higher during holidays
                },
                {
                    "name": "Luxury Items",
                    "target_margin": 200.0,
                    "min_stock": 5,
                    "restock_threshold": 3,
                    "restock_amount": 10,
                    "seasonal_factor": 1.5  # Much higher during holidays
                },
                {
                    "name": "Clearance",
                    "target_margin": 5.0,
                    "min_stock": 200,
                    "restock_threshold": 100,
                    "restock_amount": 500,
                    "seasonal_factor": 0.8  # Lower during holidays
                }
            ]
            
            # Use provided categories if any
            if categories:
                category_list = [
                    {
                        "name": cat,
                        "target_margin": 30.0,
                        "min_stock": 50,
                        "restock_threshold": 30,
                        "restock_amount": 100,
                        "seasonal_factor": 1.0
                    } for cat in categories
                ]
            else:
                category_list = default_categories
            
            for category in category_list:
                result = self.save_category(category)
                logger.info(f"Inserted category: {category['name']}")
            
            # Insert products with more realistic patterns
            logger.info(f"Inserting {num_products} products per category...")
            products = []
            
            # Regular products for each category
            for category in category_list:
                for i in range(num_products):
                    base_price = random.uniform(100, 5000)
                    product = {
                        "name": f"{category['name']} Product {i+1}",
                        "category": category['name'],
                        "price": base_price,
                        "base_price": base_price,  # Store base price for price fluctuations
                        "quantity": random.randint(50, 200),
                        "min_stock": category['min_stock'],
                        "restock_threshold": category['restock_threshold'],
                        "restock_amount": category['restock_amount'],
                        "seasonal_factor": category['seasonal_factor'],
                        "description": f"Sample {category['name'].lower()} product {i+1}",
                        "popularity_score": random.uniform(0.1, 1.0),  # Product popularity affects sales
                        "price_elasticity": random.uniform(0.5, 2.0)   # How much price changes affect demand
                    }
                    
                    # Adjust values based on category
                    if category['name'] == "Luxury Items":
                        product['price'] = random.uniform(5000, 10000)
                        product['base_price'] = product['price']
                        product['quantity'] = random.randint(1, 5)
                        product['popularity_score'] = random.uniform(0.05, 0.3)  # Luxury items sell less frequently
                        product['price_elasticity'] = random.uniform(0.3, 0.8)   # Luxury buyers less price sensitive
                    elif category['name'] == "Clearance":
                        product['price'] = random.uniform(10, 500)
                        product['base_price'] = product['price'] * 2  # Original price before clearance
                        product['quantity'] = random.randint(500, 1000)
                        product['popularity_score'] = random.uniform(0.7, 1.0)   # Clearance items sell quickly
                        product['price_elasticity'] = random.uniform(1.5, 3.0)   # Very sensitive to price
                    
                    product_id = self.add_product(product)
                    logger.info(f"Inserted product: {product['name']}")
                    products.append(product)
            
            # Special anomaly products (always included)
            anomaly_products = [
                {
                    "name": "Zero Stock Item",
                    "category": category_list[0]['name'],
                    "price": 299.99,
                    "base_price": 299.99,
                    "quantity": 0,
                    "min_stock": 50,
                    "restock_threshold": 30,
                    "restock_amount": 100,
                    "seasonal_factor": 1.0,
                    "description": "Product with zero stock",
                    "popularity_score": 0.9,  # High demand but no stock
                    "price_elasticity": 1.0
                },
                {
                    "name": "Overstock Item",
                    "category": category_list[0]['name'],
                    "price": 19.99,
                    "base_price": 39.99,  # Original price before discount
                    "quantity": 5000,
                    "min_stock": 100,
                    "restock_threshold": 50,
                    "restock_amount": 200,
                    "seasonal_factor": 0.5,  # Low seasonal variation
                    "description": "Product with excessive stock",
                    "popularity_score": 0.2,  # Low demand causing overstock
                    "price_elasticity": 2.0   # Very sensitive to price changes
                },
                {
                    "name": "Premium Product",
                    "category": category_list[0]['name'],
                    "price": 9999.99,
                    "base_price": 9999.99,
                    "quantity": 3,
                    "min_stock": 5,
                    "restock_threshold": 2,
                    "restock_amount": 5,
                    "seasonal_factor": 2.0,  # High seasonal variation
                    "description": "Premium product with very high price",
                    "popularity_score": 0.1,  # Very exclusive
                    "price_elasticity": 0.2   # Price insensitive
                }
            ]
            
            for product in anomaly_products:
                product_id = self.add_product(product)
                logger.info(f"Inserted product: {product['name']}")
                products.append(product)
            
            # Create sample sales with more realistic patterns
            logger.info(f"Creating {num_sales} sample sales...")
            
            # Generate sales over a period
            end_date = datetime.now()
            start_date = end_date - timedelta(days=num_sales)
            dates = [start_date + timedelta(days=x) for x in range(num_sales)]
            
            # Track product quantities
            product_quantities = {p['name']: p['quantity'] for p in products}
            
            # Generate sales for each day
            for sale_index, sale_date in enumerate(dates):
                try:
                    # Apply time-based factors
                    hour = random.randint(9, 20)  # Business hours
                    is_weekend = sale_date.weekday() >= 5
                    is_holiday = self._is_holiday(sale_date)  # Helper to check holidays
                    
                    # Calculate demand multiplier based on various factors
                    time_multiplier = 1.0
                    if is_weekend:
                        time_multiplier *= 1.3  # Higher weekend sales
                    if is_holiday:
                        time_multiplier *= 1.5  # Higher holiday sales
                    if 12 <= hour <= 14 or 17 <= hour <= 19:
                        time_multiplier *= 1.2  # Peak hours
                    
                    # Select products based on their popularity and current factors
                    available_products = [p for p in products if product_quantities[p['name']] > 0]
                    if not available_products:
                        continue
                        
                    # Weight product selection by popularity and seasonal factors
                    weights = [
                        p['popularity_score'] * 
                        (p['seasonal_factor'] if is_holiday else 1.0) * 
                        time_multiplier
                        for p in available_products
                    ]
                    
                    # Normalize weights
                    total_weight = sum(weights)
                    if total_weight > 0:
                        weights = [w/total_weight for w in weights]
                    else:
                        weights = None
                    
                    # Select 1-3 products based on weights
                    num_items = random.randint(1, min(3, len(available_products)))
                    sale_products = random.choices(
                        available_products,
                        weights=weights,
                        k=num_items
                    )
                    
                    sale_data = {
                        "customer": {
                            "name": f"Customer {sale_index + 1}",
                            "email": f"customer{sale_index + 1}@example.com",
                            "phone": f"{random.randint(1000000000, 9999999999)}"
                        },
                        "items": [],
                        "total_amount": 0,
                        "payment_method": random.choice(["Cash", "Card", "UPI"]),
                        "timestamp": sale_date.replace(hour=hour),
                        "status": "completed",
                        "created_at": sale_date.replace(hour=hour),
                        "updated_at": sale_date.replace(hour=hour),
                        "is_weekend": is_weekend,
                        "is_holiday": is_holiday
                    }
                    
                    # Add items to sale
                    total_amount = 0
                    for product in sale_products:
                        # Calculate quantity based on price elasticity and time factors
                        base_quantity = random.randint(1, 3)
                        price_factor = (product['base_price'] / product['price']) ** product['price_elasticity']
                        final_quantity = min(
                            int(base_quantity * price_factor * time_multiplier),
                            product_quantities[product['name']]
                        )
                        
                        if final_quantity < 1:
                            continue
                            
                        price = float(product['price'])
                        amount = round(final_quantity * price, 2)
                        
                        sale_item = {
                            "product_id": str(product.get('_id', 'temp_id')),
                            "name": product['name'],
                            "quantity": final_quantity,
                            "price": price,
                            "amount": amount,
                            "base_price": product['base_price']
                        }
                        
                        sale_data["items"].append(sale_item)
                        total_amount += amount
                        
                        # Update available quantity
                        product_quantities[product['name']] -= final_quantity
                    
                    if sale_data["items"]:  # Only save if there are items
                        sale_data["total_amount"] = round(total_amount, 2)
                        self.save_sales_data(sale_data)
                        logger.info(f"Created sale {sale_index + 1}")
                        
                        # Random restock check for each product
                        for product in products:
                            current_quantity = product_quantities[product['name']]
                            if current_quantity <= product['restock_threshold']:
                                # 80% chance to restock when below threshold
                                if random.random() < 0.8:
                                    restock_qty = product['restock_amount']
                                    product_quantities[product['name']] += restock_qty
                                    logger.info(f"Restocked {product['name']} with {restock_qty} units")
                except Exception as e:
                    logger.error(f"Error generating sale {sale_index + 1}: {str(e)}")
                    continue
            
            logger.info("Sample data generation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error generating sample data: {str(e)}")
            raise

    def _is_holiday(self, date):
        """Helper function to determine if a date is a holiday"""
        # Major Indian holidays (simplified)
        holidays = [
            (1, 1),   # New Year
            (15, 8),  # Independence Day
            (2, 10),  # Gandhi Jayanti
            (25, 12), # Christmas
        ]
        return (date.month, date.day) in holidays 