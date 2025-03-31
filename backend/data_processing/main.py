import os
import logging
import argparse
import asyncio
from dotenv import load_dotenv

from .pipeline import DataPipeline
from .french_institutes_processor import FrenchInstituteProcessor
from .international_integration import InternationalIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_processing.log"),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()

def setup_onet_config():
    """Set up configuration for O*NET database connection"""
    return {
        "host": os.getenv("ONET_DB_HOST", "localhost"),
        "user": os.getenv("ONET_DB_USER", "root"),
        "password": os.getenv("ONET_DB_PASSWORD", ""),
        "database": os.getenv("ONET_DB_NAME", "onet")
    }

def setup_neo4j_config():
    """Set up configuration for Neo4j database connection"""
    return {
        "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        "user": os.getenv("NEO4J_USER", "neo4j"),
        "password": os.getenv("NEO4J_PASSWORD", "neo4j")
    }

async def process_onet_data():
    """Process O*NET data and load into Neo4j"""
    logging.info("Processing O*NET data...")
    
    onet_config = setup_onet_config()
    neo4j_config = setup_neo4j_config()
    
    pipeline = DataPipeline(
        onet_config=onet_config,
        bls_api_key=os.getenv("BLS_API_KEY", ""),
        neo4j_config=neo4j_config
    )
    
    try:
        await pipeline.process_all_occupations()
        logging.info("O*NET data processing completed successfully")
    except Exception as e:
        logging.error(f"Error processing O*NET data: {str(e)}")
    finally:
        pipeline.close()

def process_french_data():
    """Process French formation institute data and integrate with O*NET"""
    logging.info("Processing French data...")
    
    onet_config = setup_onet_config()
    neo4j_config = setup_neo4j_config()
    french_data_path = os.getenv("FRENCH_DATA_PATH", "./data/france-raw/formation-institutes")
    
    integration = InternationalIntegration(
        onet_config=onet_config,
        neo4j_config=neo4j_config,
        french_data_path=french_data_path
    )
    
    try:
        integration.process_french_data()
        logging.info("French data processing completed successfully")
    except Exception as e:
        logging.error(f"Error processing French data: {str(e)}")
    finally:
        integration.close()

def main():
    """Main function to run the data pipeline"""
    parser = argparse.ArgumentParser(description="Career Guidance Data Pipeline")
    parser.add_argument("--onet", action="store_true", help="Process O*NET data")
    parser.add_argument("--french", action="store_true", help="Process French data")
    parser.add_argument("--all", action="store_true", help="Process all data sources")
    
    args = parser.parse_args()
    
    if args.all or (not args.onet and not args.french):
        # Process all data sources if --all is specified or if no specific source is specified
        asyncio.run(process_onet_data())
        process_french_data()
    else:
        if args.onet:
            asyncio.run(process_onet_data())
        if args.french:
            process_french_data()
    
    logging.info("Data pipeline processing completed")

if __name__ == "__main__":
    main() 