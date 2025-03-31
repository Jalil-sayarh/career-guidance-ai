import os
import sys
import logging
import pandas as pd
import mysql.connector
from neo4j import GraphDatabase
from dotenv import load_dotenv
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("connection_tester")

# Load environment variables
load_dotenv()

class ConnectionTester:
    def __init__(self):
        self.onet_config = {
            "host": os.getenv("ONET_DB_HOST", "localhost"),
            "user": os.getenv("ONET_DB_USER", "root"),
            "password": os.getenv("ONET_DB_PASSWORD", ""),
            "database": os.getenv("ONET_DB_NAME", "onet")
        }
        
        self.neo4j_config = {
            "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            "user": os.getenv("NEO4J_USER", "neo4j"),
            "password": os.getenv("NEO4J_PASSWORD", "neo4j")
        }
        
        self.french_data_path = os.getenv("FRENCH_DATA_PATH", "./data/france-raw/formation-institutes")
    
    def test_mysql_connection(self):
        """Test connection to the MySQL O*NET database"""
        logger.info("Testing MySQL connection...")
        try:
            connection = mysql.connector.connect(**self.onet_config)
            cursor = connection.cursor(dictionary=True)
            
            # Test a basic query to validate schema
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            
            if tables:
                logger.info(f"Successfully connected to MySQL. Found {len(tables)} tables.")
                
                # Sample a few key tables to verify
                key_tables = ["occupation_data", "skills", "abilities", "interests"]
                for table in key_tables:
                    try:
                        cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
                        count = cursor.fetchone()["count"]
                        logger.info(f"Table '{table}' contains {count} records")
                    except Exception as e:
                        logger.warning(f"Table '{table}' might not exist: {str(e)}")
                
                return True
            else:
                logger.error("Connected to MySQL but no tables found. Is the database populated?")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to MySQL: {str(e)}")
            return False
        finally:
            try:
                cursor.close()
                connection.close()
            except:
                pass
    
    def test_neo4j_connection(self):
        """Test connection to Neo4j database"""
        logger.info("Testing Neo4j connection...")
        try:
            driver = GraphDatabase.driver(
                self.neo4j_config["uri"],
                auth=(self.neo4j_config["user"], self.neo4j_config["password"])
            )
            
            with driver.session() as session:
                # Run a simple query to verify connection
                result = session.run("MATCH (n) RETURN count(n) as count")
                count = result.single()["count"]
                logger.info(f"Successfully connected to Neo4j. Database contains {count} nodes.")
                
                # Test optional Neo4j plugin availability
                try:
                    session.run("CALL gds.list()")
                    logger.info("Graph Data Science Library is available")
                except Exception:
                    logger.warning("Graph Data Science Library not installed - this is optional but recommended")
                
                try:
                    session.run("RETURN vectorize('test')")
                    logger.info("Vector capabilities are available")
                except Exception:
                    logger.warning("Vector capabilities not available - this is optional but recommended for semantic search")
                
                return True
        
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            return False
        finally:
            try:
                driver.close()
            except:
                pass
    
    def test_french_data_access(self):
        """Test access to French data files"""
        logger.info("Testing access to French data files...")
        try:
            required_files = ["public_ofs.csv", "ListeNSF.xlsx"]
            
            for file in required_files:
                file_path = os.path.join(self.french_data_path, file)
                if not os.path.exists(file_path):
                    logger.error(f"Required file not found: {file_path}")
                    return False
                
                logger.info(f"Found file: {file}")
            
            # Test loading CSV
            csv_path = os.path.join(self.french_data_path, "public_ofs.csv")
            try:
                df = pd.read_csv(csv_path, sep=';', encoding='utf-8', nrows=5)
                logger.info(f"Successfully read CSV. Sample columns: {', '.join(df.columns[:5])}")
            except Exception as e:
                logger.error(f"Error reading CSV file: {str(e)}")
                return False
                
            # Test loading Excel
            excel_path = os.path.join(self.french_data_path, "ListeNSF.xlsx")
            try:
                df = pd.read_excel(excel_path, nrows=5)
                logger.info(f"Successfully read Excel. Sample columns: {', '.join(df.columns[:5])}")
            except Exception as e:
                logger.error(f"Error reading Excel file: {str(e)}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error testing French data access: {str(e)}")
            return False
    
    def run_all_tests(self):
        """Run all connection tests"""
        results = {
            "mysql": self.test_mysql_connection(),
            "neo4j": self.test_neo4j_connection(),
            "french_data": self.test_french_data_access()
        }
        
        # Print summary
        logger.info("\n--- TEST RESULTS SUMMARY ---")
        all_passed = True
        for test, passed in results.items():
            status = "PASSED" if passed else "FAILED"
            if not passed:
                all_passed = False
            logger.info(f"{test.upper()}: {status}")
        
        if all_passed:
            logger.info("\nALL TESTS PASSED! You can proceed with the data pipeline.")
        else:
            logger.warning("\nSome tests failed. Please fix the issues before proceeding.")
        
        return all_passed

def main():
    parser = argparse.ArgumentParser(description="Test connections for Career Guidance data pipeline")
    parser.add_argument("--mysql", action="store_true", help="Test only MySQL connection")
    parser.add_argument("--neo4j", action="store_true", help="Test only Neo4j connection")
    parser.add_argument("--french", action="store_true", help="Test only French data access")
    
    args = parser.parse_args()
    tester = ConnectionTester()
    
    if args.mysql:
        tester.test_mysql_connection()
    elif args.neo4j:
        tester.test_neo4j_connection()
    elif args.french:
        tester.test_french_data_access()
    else:
        tester.run_all_tests()

if __name__ == "__main__":
    main() 