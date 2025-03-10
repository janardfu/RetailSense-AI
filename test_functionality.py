import unittest
from database import MongoDB
from datetime import datetime, timedelta
import pandas as pd
import logging
from bson import ObjectId
import json
from pymongo.errors import ConnectionFailure, OperationFailure

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestRetailSense(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test database connection"""
        try:
            cls.db = MongoDB()
            logger.info("Database connection established successfully")
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during database setup: {str(e)}")
            raise

    def setUp(self):
        """Set up test data before each test"""
        try:
            # Generate fresh sample data for each test
            self.db.generate_sample_data()
            logger.info("Test data generated successfully")
        except OperationFailure as e:
            logger.error(f"Database operation failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to generate test data: {str(e)}")
            raise

    def test_1_database_connection(self):
        """Test database connection"""
        try:
            # Test valid connection
            self.assertTrue(self.db.validate_connection())
            
            # Test invalid connection
            with self.assertRaises(ConnectionFailure):
                invalid_db = MongoDB(uri="mongodb://invalid:27017", timeout=1)
                invalid_db.validate_connection()
                
            logger.info("Database connection tests passed")
            
        except Exception as e:
            logger.error(f"Database connection test failed: {str(e)}")
            raise

    def test_2_generate_sample_data(self):
        """Test sample data generation"""
        try:
            # Test valid data generation
            self.assertTrue(self.db.generate_sample_data(num_products=5, num_sales=3))
            
            # Test invalid parameters
            with self.assertRaises(ValueError):
                self.db.generate_sample_data(num_products=-1)
            
            with self.assertRaises(ValueError):
                self.db.generate_sample_data(num_sales=-1)
            
            with self.assertRaises(ValueError):
                self.db.generate_sample_data(categories="invalid")
            
            logger.info("Sample data generation tests passed")
            
        except Exception as e:
            logger.error(f"Sample data generation test failed: {str(e)}")
            raise

    def test_3_inventory_management(self):
        """Test inventory management functions"""
        try:
            # Test valid product update
            product_id = self.db.add_product({
                "name": "Test Product",
                "category": "Test Category",
                "price": 100,
                "quantity": 10,
                "min_stock": 5
            })
            
            self.assertTrue(self.db.update_product_quantity(product_id, 15))
            
            # Test invalid product ID
            with self.assertRaises(ValueError):
                self.db.update_product_quantity("invalid_id", 10)
            
            # Test negative quantity with allow_negative=False
            with self.assertRaises(ValueError):
                self.db.update_product_quantity(product_id, -1, allow_negative=False)
            
            logger.info("Inventory management tests passed")
            
        except Exception as e:
            logger.error(f"Inventory management test failed: {str(e)}")
            raise

    def test_4_sales_management(self):
        """Test sales management functions"""
        try:
            # Test normal sales operations
            sales = self.db.get_sales_data()
            self.assertIsNotNone(sales)
            self.assertGreater(len(sales), 0)
            
            # Test sales data structure
            for sale in sales:
                required_fields = ['customer', 'items', 'total_amount', 'payment_method']
                for field in required_fields:
                    self.assertIn(field, sale)
                
                # Validate customer data
                self.assertIn('name', sale['customer'])
                self.assertIn('phone', sale['customer'])
                
                # Validate items
                self.assertGreater(len(sale['items']), 0)
                for item in sale['items']:
                    self.assertIn('product_id', item)
                    self.assertIn('name', item)
                    self.assertIn('quantity', item)
                    self.assertIn('price', item)
                    self.assertIn('amount', item)
                
                # Validate total amount
                calculated_total = sum(item['amount'] for item in sale['items'])
                self.assertAlmostEqual(calculated_total, sale['total_amount'])
            
            # Test invalid sales data
            invalid_sales = [
                {
                    'customer': {'name': 'Test Customer'},
                    'items': [],
                    'total_amount': 0,
                    'payment_method': 'Cash'
                },
                {
                    'customer': {'name': 'Test Customer'},
                    'items': [{'product_id': 'invalid_id', 'quantity': 1}],
                    'total_amount': 100,
                    'payment_method': 'Invalid'
                },
                {
                    'customer': {'name': 'Test Customer'},
                    'items': [{'product_id': 'valid_id', 'quantity': -1}],
                    'total_amount': 100,
                    'payment_method': 'Cash'
                }
            ]
            
            for invalid_sale in invalid_sales:
                with self.assertRaises(ValueError):
                    self.db.save_sales_data(invalid_sale)
            
            logger.info("Sales management tests passed")
        except Exception as e:
            logger.error(f"Sales management test failed: {str(e)}")
            raise

    def test_5_ai_features(self):
        """Test AI-powered features"""
        try:
            # Test valid anomaly detection
            anomalies = self.db.detect_anomalies()
            self.assertIsInstance(anomalies, list)
            
            # Test invalid parameters
            with self.assertRaises(ValueError):
                self.db.calculate_safety_stock("", service_level=2.0)
            
            with self.assertRaises(ValueError):
                self.db.optimize_inventory(holding_cost_rate=-0.1)
            
            logger.info("AI features tests passed")
            
        except Exception as e:
            logger.error(f"AI features test failed: {str(e)}")
            raise

    def test_6_data_validation(self):
        """Test data validation"""
        try:
            # Test valid product data
            product_id = self.db.add_product({
                "name": "Test Product",
                "category": "Test Category",
                "price": 100,
                "quantity": 10,
                "min_stock": 5
            })
            
            product = self.db.get_product(product_id)
            self.assertGreaterEqual(product['quantity'], 0)
            
            # Test invalid product data
            with self.assertRaises(ValueError):
                self.db.add_product({
                    "name": "Invalid Product",
                    "category": "Test Category",
                    "price": -100,
                    "quantity": 10,
                    "min_stock": 5
                })
            
            logger.info("Data validation tests passed")
            
        except Exception as e:
            logger.error(f"Data validation test failed: {str(e)}")
            raise

    def test_7_performance_metrics(self):
        """Test performance metrics"""
        try:
            # Test valid collection stats
            stats = self.db.get_collection_stats()
            self.assertIsInstance(stats, dict)
            
            # Test invalid collection
            with self.assertRaises(ValueError):
                self.db.get_collection_stats(collection="invalid_collection")
            
            logger.info("Performance metrics tests passed")
            
        except Exception as e:
            logger.error(f"Performance metrics test failed: {str(e)}")
            raise

    def test_8_error_handling(self):
        """Test error handling"""
        try:
            # Test invalid product ID
            invalid_product = self.db.get_product("invalid_id")
            self.assertIsNone(invalid_product)
            
            # Test invalid category
            invalid_category_count = self.db.get_product_count_by_category("invalid_category")
            self.assertEqual(invalid_category_count, 0)
            
            # Test invalid sales data
            invalid_sale = {
                'customer': {'name': 'Test Customer'},
                'items': [],
                'total_amount': 0,
                'payment_method': 'Cash'
            }
            with self.assertRaises(ValueError):
                self.db.save_sales_data(invalid_sale)
            
            # Test invalid inventory update
            with self.assertRaises(ValueError):
                self.db.update_product_quantity("invalid_id", -1)
            
            # Test database connection errors
            with self.assertRaises(ConnectionFailure):
                invalid_db = MongoDB(uri="mongodb://invalid:27017")
                invalid_db.get_all_products()
            
            # Test operation errors
            with self.assertRaises(OperationFailure):
                self.db.db.command("invalid_command")
            
            logger.info("Error handling tests passed")
        except Exception as e:
            logger.error(f"Error handling test failed: {str(e)}")
            raise

    def test_9_data_integrity(self):
        """Test data integrity"""
        try:
            # Generate test data
            self.db.generate_sample_data()
            logger.info("Test data generated successfully")
            
            # Get a valid product ID from the database
            product = self.db.products.find_one({})
            if not product:
                raise ValueError("No products found in database")
            valid_product_id = str(product['_id'])
            
            # Test negative quantity (should be set to 0)
            self.db.update_product_quantity(valid_product_id, -1, allow_negative=True)
            updated_product = self.db.products.find_one({'_id': ObjectId(valid_product_id)})
            assert updated_product['quantity'] == 0, "Negative quantity should be set to 0"
            
            # Test invalid product ID
            with self.assertRaises(ValueError):
                self.db.update_product_quantity("invalid_id", 10)
            
            # Test non-numeric quantity
            with self.assertRaises(TypeError):
                self.db.update_product_quantity(valid_product_id, "invalid")
            
        except Exception as e:
            logger.error(f"Data integrity test failed: {str(e)}")
            raise

    def test_10_api_endpoints(self):
        """Test API endpoints"""
        try:
            # Test valid product retrieval
            product_id = self.db.add_product({
                "name": "Test Product",
                "category": "Test Category",
                "price": 100,
                "quantity": 10,
                "min_stock": 5
            })
            
            product = self.db.get_product(product_id, raise_error=True)
            self.assertEqual(str(product['_id']), str(product_id))
            
            # Test invalid product ID
            with self.assertRaises(ValueError):
                self.db.get_product("invalid_id", raise_error=True)
            
            logger.info("API endpoints tests passed")
            
        except Exception as e:
            logger.error(f"API endpoints test failed: {str(e)}")
            raise

if __name__ == '__main__':
    unittest.main(verbosity=2) 