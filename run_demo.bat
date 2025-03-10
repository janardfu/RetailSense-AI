@echo off
echo Starting AI-Powered Inventory Management System...

REM Activate virtual environment
call venv\Scripts\activate

REM Check MongoDB connection
echo Checking MongoDB connection...
python -c "from pymongo import MongoClient; client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=2000); client.server_info()" >nul 2>&1
if errorlevel 1 (
    echo ERROR: MongoDB is not running! Please start MongoDB service.
    echo You can start MongoDB using the following steps:
    echo 1. Open Command Prompt as Administrator
    echo 2. Run: net start MongoDB
    exit /b 1
)

REM Set environment variables if not already set
if not defined CURRENCY_SYMBOL set CURRENCY_SYMBOL=₹
if not defined TIMEZONE set TIMEZONE=Asia/Kolkata

REM Clear cache and temporary files
echo Cleaning up cache...
if exist .streamlit\cache rmdir /s /q .streamlit\cache
if exist __pycache__ rmdir /s /q __pycache__

REM Start the application
echo Starting application...
streamlit run app.py

REM Keep the window open if there's an error
if errorlevel 1 (
    echo An error occurred while running the application.
    pause
)

REM Deactivate virtual environment on exit
deactivate 