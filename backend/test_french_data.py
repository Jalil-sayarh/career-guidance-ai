import os
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("french_data_test")

def analyze_french_data(file_path: str):
    """Analyze a French data file and print its structure"""
    logger.info(f"Analyzing file: {file_path}")
    
    try:
        # Read the file based on its extension
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, encoding='utf-8', nrows=5)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path, nrows=5)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Print file information
        print(f"\nFile: {os.path.basename(file_path)}")
        print(f"Shape: {df.shape}")
        print("\nColumns:")
        for col in df.columns:
            print(f"- {col} (type: {df[col].dtype})")
        
        print("\nSample data:")
        print(df.head())
        
        # Print summary statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            print("\nSummary statistics for numeric columns:")
            print(df[numeric_cols].describe())
        
    except Exception as e:
        logger.error(f"Error analyzing file {file_path}: {str(e)}")

def main():
    """Main function to test French data parsing"""
    french_data_dir = "../data/countries/france/formation-institutes"
    
    if not os.path.exists(french_data_dir):
        logger.error(f"Directory not found: {french_data_dir}")
        return
    
    for file in os.listdir(french_data_dir):
        if file.endswith(('.csv', '.xlsx')):
            file_path = os.path.join(french_data_dir, file)
            analyze_french_data(file_path)

if __name__ == "__main__":
    main() 