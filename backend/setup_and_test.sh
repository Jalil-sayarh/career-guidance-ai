#!/bin/bash

# Career Guidance AI - Setup and Test Script
echo "Career Guidance AI - Setup and Test Script"
echo "==========================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not found. Please install Python 3 and try again."
    exit 1
fi

# Check if .env file exists, if not create from template
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.template .env
    echo "Please edit the .env file with your configuration details."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install requirements
    echo "Installing dependencies..."
    pip install -r requirements.txt
else
    # Activate virtual environment
    source venv/bin/activate
fi

# Run connection tests
echo "Testing database connections..."
python -m data_processing.test_connections

# Ask if user wants to create test data
read -p "Do you want to create and validate test data in Neo4j? (y/N): " create_test_data
if [[ $create_test_data =~ ^[Yy]$ ]]; then
    echo "Creating and validating test data..."
    python -m data_processing.test_data
fi

# Ask if user wants to try the full pipeline with a sample
read -p "Do you want to run the pipeline with a small sample? (y/N): " run_pipeline
if [[ $run_pipeline =~ ^[Yy]$ ]]; then
    echo "Running pipeline with a sample..."
    python -m data_processing.main --sample
fi

echo "Setup and testing complete!" 