@echo off
REM Career Guidance AI - Setup and Test Script
echo Career Guidance AI - Setup and Test Script
echo ===========================================

REM Check if Python is installed
python --version 2>NUL
if %ERRORLEVEL% NEQ 0 (
    echo Python is required but not found. Please install Python and try again.
    exit /b 1
)

REM Check if .env file exists, if not create from template
if not exist .env (
    echo Creating .env file from template...
    copy .env.template .env
    echo Please edit the .env file with your configuration details.
    exit /b 1
)

REM Check if virtual environment exists
if not exist venv\ (
    echo Creating virtual environment...
    python -m venv venv
    
    REM Activate virtual environment
    call venv\Scripts\activate
    
    REM Install requirements
    echo Installing dependencies...
    pip install -r requirements.txt
) else (
    REM Activate virtual environment
    call venv\Scripts\activate
)

REM Run connection tests
echo Testing database connections...
python -m data_processing.test_connections

REM Ask if user wants to create test data
set /p create_test_data="Do you want to create and validate test data in Neo4j? (y/N): "
if /i "%create_test_data%"=="y" (
    echo Creating and validating test data...
    python -m data_processing.test_data
)

REM Ask if user wants to try the full pipeline with a sample
set /p run_pipeline="Do you want to run the pipeline with a small sample? (y/N): "
if /i "%run_pipeline%"=="y" (
    echo Running pipeline with a small sample...
    python -m data_processing.main --sample
)

echo Setup and testing complete!
pause 