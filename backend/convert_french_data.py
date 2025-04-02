import os
import pandas as pd
import sqlite3
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("french_data_converter")

class FrenchDataConverter:
    def __init__(self, data_dir: str, db_path: str):
        """Initialize the converter with data directory and database path"""
        self.data_dir = data_dir
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        
    def connect_db(self):
        """Connect to SQLite database"""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        logger.info(f"Connected to database: {self.db_path}")
        
    def close_db(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
            
    def convert_public_ofs(self, file_path: str):
        """Convert public_ofs_v2.xlsx to SQL"""
        logger.info(f"Converting {file_path} to SQL")
        
        # Read Excel file
        df = pd.read_excel(file_path)
        
        # Clean column names - make SQL compatible
        def clean_column_name(col):
            # Replace French special characters
            replacements = {
                'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
                'à': 'a', 'â': 'a', 'ä': 'a',
                'î': 'i', 'ï': 'i',
                'ô': 'o', 'ö': 'o',
                'ù': 'u', 'û': 'u', 'ü': 'u',
                'ç': 'c',
                ' ': '_', '-': '_', '/': '_', '(': '', ')': '', '.': '_',
                "'": '_', '"': '_'
            }
            col = col.lower()
            for old, new in replacements.items():
                col = col.replace(old, new)
            return col
        
        df.columns = [clean_column_name(col) for col in df.columns]
        
        # Remove duplicate underscores and trailing numbers
        df.columns = [col.strip('_0123456789') for col in df.columns]
        df.columns = [col[:-1] if col.endswith('_') else col for col in df.columns]
        
        # Ensure unique column names
        seen = set()
        new_cols = []
        for col in df.columns:
            base_col = col
            i = 1
            while col in seen:
                col = f"{base_col}_{i}"
                i += 1
            seen.add(col)
            new_cols.append(col)
        df.columns = new_cols
        
        # Drop existing table if it exists
        self.cursor.execute("DROP TABLE IF EXISTS public_ofs")
        
        # Create table
        columns = []
        for col in df.columns:
            # Determine SQL type based on data type
            if pd.api.types.is_numeric_dtype(df[col]):
                sql_type = 'REAL'
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                sql_type = 'DATETIME'
            else:
                sql_type = 'TEXT'
            columns.append(f'"{col}" {sql_type}')
            
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS public_ofs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            {', '.join(columns)}
        )
        """
        
        self.cursor.execute(create_table_sql)
        
        # Insert data
        for _, row in df.iterrows():
            # Convert NaN to None and handle special characters
            values = []
            for val in row:
                if pd.isna(val):
                    values.append(None)
                elif isinstance(val, (int, float)):
                    values.append(val)
                else:
                    values.append(str(val))
            
            placeholders = ','.join(['?' for _ in values])
            insert_sql = f'INSERT INTO public_ofs ("{str('","'.join(df.columns))}") VALUES ({placeholders})'
            try:
                self.cursor.execute(insert_sql, tuple(values))
            except Exception as e:
                logger.error(f"Error inserting row: {e}")
                logger.error(f"SQL: {insert_sql}")
                logger.error(f"Values: {values}")
                raise
            
        self.conn.commit()
        logger.info(f"Converted {len(df)} rows from public_ofs_v2.xlsx")
        
    def convert_liste_nsf(self, file_path: str):
        """Convert ListeNSF.xlsx to SQL"""
        logger.info(f"Converting {file_path} to SQL")
        
        # Read Excel file
        df = pd.read_excel(file_path)
        
        # Clean column names
        df.columns = ['code', 'specialty']
        
        # Drop existing table if it exists
        self.cursor.execute("DROP TABLE IF EXISTS nsf_codes")
        
        # Create table
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS nsf_codes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code INTEGER,
            specialty TEXT
        )
        """
        
        self.cursor.execute(create_table_sql)
        
        # Insert data
        for _, row in df.iterrows():
            if pd.notna(row['code']) and pd.notna(row['specialty']):
                insert_sql = "INSERT INTO nsf_codes (code, specialty) VALUES (?, ?)"
                self.cursor.execute(insert_sql, (int(row['code']), row['specialty']))
                
        self.conn.commit()
        logger.info(f"Converted {len(df)} rows from ListeNSF.xlsx")
        
    def convert_dessin_enregistrement(self, file_path: str):
        """Convert Dessin_Enregistrement_ListeOF.xlsx to SQL"""
        logger.info(f"Converting {file_path} to SQL")
        
        # Read Excel file
        df = pd.read_excel(file_path)
        
        # Clean column names
        df.columns = ['field_name', 'description']
        
        # Drop existing table if it exists
        self.cursor.execute("DROP TABLE IF EXISTS field_descriptions")
        
        # Create table
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS field_descriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            field_name TEXT,
            description TEXT
        )
        """
        
        self.cursor.execute(create_table_sql)
        
        # Insert data
        for _, row in df.iterrows():
            if pd.notna(row['field_name']) and pd.notna(row['description']):
                insert_sql = "INSERT INTO field_descriptions (field_name, description) VALUES (?, ?)"
                self.cursor.execute(insert_sql, (row['field_name'], row['description']))
                
        self.conn.commit()
        logger.info(f"Converted {len(df)} rows from Dessin_Enregistrement_ListeOF.xlsx")
        
    def convert_all_files(self):
        """Convert all French data files to SQL"""
        try:
            self.connect_db()
            
            # Convert each file
            formation_dir = os.path.join(self.data_dir, "france", "formation-institutes")
            
            for file in os.listdir(formation_dir):
                file_path = os.path.join(formation_dir, file)
                
                if file == "public_ofs_v2.xlsx":
                    self.convert_public_ofs(file_path)
                elif file == "ListeNSF.xlsx":
                    self.convert_liste_nsf(file_path)
                elif file == "Dessin_Enregistrement_ListeOF.xlsx":
                    self.convert_dessin_enregistrement(file_path)
                else:
                    logger.warning(f"Skipping unsupported file: {file}")
                    
            logger.info("All files converted successfully")
            
        except Exception as e:
            logger.error(f"Error converting files: {str(e)}")
            raise
        finally:
            self.close_db()

def main():
    """Main function to convert French data files"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert French Excel files to SQL")
    parser.add_argument("--data-dir", required=True, help="Directory containing country data")
    parser.add_argument("--db-path", required=True, help="Path to SQLite database file")
    
    args = parser.parse_args()
    
    converter = FrenchDataConverter(args.data_dir, args.db_path)
    converter.convert_all_files()

if __name__ == "__main__":
    main() 