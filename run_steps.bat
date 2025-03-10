@echo off
echo Setting up AI-Powered Inventory Management System...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed! Please install Python 3.8 or higher.
    exit /b 1
)

REM Check if MongoDB is installed and running
mongod --version >nul 2>&1
if errorlevel 1 (
    echo MongoDB is not installed! Please install MongoDB 4.4 or higher.
    exit /b 1
)

REM Create and activate virtual environment
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate

REM Install required packages
echo Installing required packages...
pip install -r requirements.txt

REM Create necessary directories
if not exist "models" mkdir models
if not exist "data" mkdir data

REM Set environment variables
echo Setting up environment variables...
set MONGODB_URI=mongodb://localhost:27017/
set DB_NAME=inventory_management
set CURRENCY_SYMBOL=₹
set TIMEZONE=Asia/Kolkata

REM Initialize database with categories
echo Initializing database with categories...
python -c "from database import MongoDB; db = MongoDB(); [db.save_category({'name': cat}) for cat in ['Electronics', 'Mobile Accessories', 'Home Appliances', 'Computer Parts', 'Audio Devices']]"

echo Setup completed successfully!
python -m streamlit run app.py
echo Application Started! 