import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException, WebDriverException
import subprocess
import os
import sys
import signal
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RetailSenseDemo:
    def __init__(self):
        self.streamlit_process = None
        self.driver = None
        self.wait = None

    def start_streamlit(self):
        """Start the Streamlit application"""
        logger.info("Starting RetailSense application...")
        try:
            self.streamlit_process = subprocess.Popen(
                ["streamlit", "run", "app.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            time.sleep(15)  # Wait longer for the application to start
            logger.info("Streamlit process started successfully")
        except Exception as e:
            logger.error(f"Failed to start Streamlit: {str(e)}")
            raise

    def stop_streamlit(self):
        """Stop the Streamlit application"""
        if self.streamlit_process:
            try:
                if sys.platform == 'win32':
                    subprocess.call(['taskkill', '/F', '/T', '/PID', str(self.streamlit_process.pid)])
                else:
                    os.killpg(os.getpgid(self.streamlit_process.pid), signal.SIGTERM)
                logger.info("Streamlit process stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping Streamlit: {str(e)}")

    def setup_driver(self):
        """Initialize the Chrome WebDriver"""
        logger.info("Initializing Chrome WebDriver...")
        try:
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service)
            self.driver.maximize_window()
            self.wait = WebDriverWait(self.driver, 20)  # Increased timeout
            logger.info("Chrome WebDriver initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize WebDriver: {str(e)}")
            raise

    def navigate_to(self, section):
        """Navigate to a specific section"""
        logger.info(f"Navigating to {section}...")
        try:
            # First try the exact match
            xpath = f"//div[normalize-space(text())='{section}']"
            try:
                link = self.wait.until(EC.element_to_be_clickable((By.XPATH, xpath)))
            except TimeoutException:
                # If exact match fails, try contains
                xpath = f"//div[contains(text(), '{section}')]"
                link = self.wait.until(EC.element_to_be_clickable((By.XPATH, xpath)))
            
            link.click()
            time.sleep(3)
            logger.info(f"Successfully navigated to {section}")
        except Exception as e:
            logger.error(f"Failed to navigate to {section}: {str(e)}")
            raise

    def wait_for_element(self, by, value, timeout=10):
        """Wait for an element to be present and visible"""
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by, value))
            )
            return element
        except TimeoutException:
            logger.error(f"Timeout waiting for element: {value}")
            return None

    def safe_click(self, element):
        """Safely click an element with retry"""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                element.click()
                return True
            except Exception as e:
                if attempt == max_attempts - 1:
                    logger.error(f"Failed to click element after {max_attempts} attempts: {str(e)}")
                    return False
                time.sleep(1)

    def demo_dashboard(self):
        """Demonstrate Dashboard features"""
        print("\n=== Demonstrating Dashboard ===")
        self.navigate_to("Dashboard")
        
        # View KPIs
        print("Viewing Key Performance Indicators...")
        time.sleep(3)
        
        # Examine sales trends
        print("Examining sales trends...")
        time.sleep(3)
        
        # Check stock alerts
        print("Checking stock alerts...")
        time.sleep(3)

    def demo_product_management(self):
        """Demonstrate Product Management features"""
        print("\n=== Demonstrating Product Management ===")
        self.navigate_to("Products")
        
        # Add new product
        print("Adding new products...")
        products = [
            {"name": "Gaming Laptop", "category": "Electronics", "price": "1299.99", "stock": "50"},
            {"name": "Wireless Mouse", "category": "Accessories", "price": "29.99", "stock": "200"},
            {"name": "USB-C Hub", "category": "Accessories", "price": "49.99", "stock": "150"}
        ]
        
        for product in products:
            # Click "Add New Product" tab
            add_product_tab = self.wait.until(EC.element_to_be_clickable((By.XPATH, "//div[contains(text(), 'Add New Product')]")))
            add_product_tab.click()
            time.sleep(1)
            
            # Fill product details
            self.driver.find_element(By.XPATH, "//input[@aria-label='Product Name']").send_keys(product["name"])
            self.driver.find_element(By.XPATH, "//input[@aria-label='Base Price']").send_keys(product["price"])
            self.driver.find_element(By.XPATH, "//input[@aria-label='Initial Stock']").send_keys(product["stock"])
            
            # Submit form
            submit_button = self.driver.find_element(By.XPATH, "//button[contains(text(), 'Add Product')]")
            submit_button.click()
            time.sleep(2)
        
        # Search and filter products
        print("Demonstrating search and filter functionality...")
        search_box = self.driver.find_element(By.XPATH, "//input[@placeholder='Search products...']")
        search_box.send_keys("Laptop")
        time.sleep(2)
        search_box.clear()
        time.sleep(1)

    def demo_category_management(self):
        """Demonstrate Category Management features"""
        print("\n=== Demonstrating Category Management ===")
        self.navigate_to("Categories")
        
        # Add new categories
        print("Adding new categories...")
        categories = [
            {"name": "Electronics", "description": "Electronic devices and gadgets", "margin": "35"},
            {"name": "Accessories", "description": "Computer and phone accessories", "margin": "45"},
            {"name": "Software", "description": "Digital software licenses", "margin": "60"}
        ]
        
        for category in categories:
            # Click "Add New Category" tab
            add_category_tab = self.wait.until(EC.element_to_be_clickable((By.XPATH, "//div[contains(text(), 'Add New Category')]")))
            add_category_tab.click()
            time.sleep(1)
            
            # Fill category details
            self.driver.find_element(By.XPATH, "//input[@aria-label='Category Name']").send_keys(category["name"])
            self.driver.find_element(By.XPATH, "//textarea[@aria-label='Description']").send_keys(category["description"])
            margin_input = self.driver.find_element(By.XPATH, "//input[@aria-label='Target Margin (%)']")
            margin_input.send_keys(Keys.CONTROL + "a")  # Select all
            margin_input.send_keys(Keys.DELETE)         # Delete existing value
            margin_input.send_keys(category["margin"])
            
            # Submit form
            submit_button = self.driver.find_element(By.XPATH, "//button[contains(text(), 'Add Category')]")
            submit_button.click()
            time.sleep(2)

    def demo_inventory_management(self):
        """Demonstrate Inventory Management features"""
        print("\n=== Demonstrating Inventory Management ===")
        self.navigate_to("Inventory")
        
        # View current stock
        print("Viewing current stock levels...")
        time.sleep(3)
        
        # Add inventory entries
        print("Adding new inventory entries...")
        entries = [
            {
                "product": "Gaming Laptop",
                "quantity": "10",
                "price": "1199.99",
                "supplier": "TechSupply Inc",
                "batch": "GL2024-001"
            },
            {
                "product": "Wireless Mouse",
                "quantity": "50",
                "price": "24.99",
                "supplier": "AccessoryWorld",
                "batch": "WM2024-001"
            }
        ]
        
        for entry in entries:
            # Click "Add Entry" tab
            add_entry_tab = self.wait.until(EC.element_to_be_clickable((By.XPATH, "//div[contains(text(), 'Add Entry')]")))
            add_entry_tab.click()
            time.sleep(1)
            
            # Fill inventory entry details
            self.driver.find_element(By.XPATH, "//input[@aria-label='Product Name']").send_keys(entry["product"])
            self.driver.find_element(By.XPATH, "//input[@aria-label='Quantity']").send_keys(entry["quantity"])
            self.driver.find_element(By.XPATH, "//input[@aria-label='Unit Price']").send_keys(entry["price"])
            self.driver.find_element(By.XPATH, "//input[@aria-label='Supplier']").send_keys(entry["supplier"])
            self.driver.find_element(By.XPATH, "//input[@aria-label='Batch Number']").send_keys(entry["batch"])
            
            # Submit form
            submit_button = self.driver.find_element(By.XPATH, "//button[contains(text(), 'Add Entry')]")
            submit_button.click()
            time.sleep(2)
        
        # Demonstrate bulk upload
        print("Demonstrating bulk upload functionality...")
        time.sleep(2)

    def demo_analytics(self):
        """Demonstrate Analytics features"""
        print("\n=== Demonstrating Analytics ===")
        self.navigate_to("Analytics")
        
        # View sales forecasts
        print("Viewing sales forecasts...")
        time.sleep(3)
        
        # Check product trends
        print("Analyzing product trends...")
        time.sleep(3)
        
        # View performance metrics
        print("Examining performance metrics...")
        time.sleep(3)
        
        # Demonstrate different forecast models
        print("Showing different forecast models...")
        time.sleep(3)

    def run_demo(self):
        """Run the complete RetailSense demonstration"""
        logger.info("Starting RetailSense Automated Demo")
        
        try:
            self.start_streamlit()
            self.setup_driver()
            
            # Navigate to the application
            logger.info("Opening RetailSense application...")
            self.driver.get("http://localhost:8501")
            time.sleep(10)  # Wait longer for initial load
            
            # Run through all demonstrations
            self.demo_dashboard()
            self.demo_category_management()
            self.demo_product_management()
            self.demo_inventory_management()
            self.demo_analytics()
            
            # Return to dashboard
            logger.info("Completing demo...")
            self.navigate_to("Dashboard")
            time.sleep(3)
            
            logger.info("Demo completed successfully!")
            
        except TimeoutException as e:
            logger.error(f"Timeout error: {str(e)}")
        except WebDriverException as e:
            logger.error(f"WebDriver error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during demo: {str(e)}")
        finally:
            # Clean up
            if self.driver:
                try:
                    self.driver.quit()
                    logger.info("WebDriver closed successfully")
                except Exception as e:
                    logger.error(f"Error closing WebDriver: {str(e)}")
            self.stop_streamlit()

if __name__ == "__main__":
    try:
        demo = RetailSenseDemo()
        demo.run_demo()
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1) 