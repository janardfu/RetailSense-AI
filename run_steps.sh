#!/bin/bash

# Enable error reporting
set -e

echo "Setting up AI-Powered Inventory Management System..."

# Check if Python is installed
echo "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "Python is not installed! Please install Python 3.8 or higher."
    exit 1
fi

# Check if MongoDB is installed and running
echo "Checking MongoDB installation..."
if ! command -v mongod &> /dev/null; then
    echo "MongoDB is not installed! Please install MongoDB 4.4 or higher."
    exit 1
fi

# Create and activate virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install required packages
echo "Installing required packages..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p models data

# Set environment variables
echo "Setting up environment variables..."
export MONGODB_URI="mongodb://localhost:27017/"
export DB_NAME="inventory_management"
export CURRENCY_SYMBOL="₹"
export TIMEZONE="Asia/Kolkata"

# Initialize database with categories
echo "Initializing database with categories..."
python3 -c "from database import MongoDB; db = MongoDB(); [db.save_category({'name': cat}) for cat in ['Electronics', 'Mobile Accessories', 'Home Appliances', 'Computer Parts', 'Audio Devices']]"

# Make the script executable
chmod +x run_steps.sh

echo "Setup completed successfully!"
python3 -m streamlit run app.py
echo "Application Started!" 