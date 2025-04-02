import os
import logging
import time
from rag_engine import CareerGuidanceRAG
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("vector_store_generator")

def main():
    """Generate and save the vector store for ONET data"""
    # Get paths from environment variables
    sql_dir = os.getenv("ONET_SQL_DIR", "./data/documentation/ONET")
    vector_store_path = os.getenv("VECTOR_STORE_PATH", "./vector_store")
    
    # Create RAG engine
    rag_engine = CareerGuidanceRAG(sql_dir)
    
    # Load data and create vector store
    logger.info("Loading ONET data and creating vector store...")
    start_time = time.time()
    rag_engine.load_data()
    end_time = time.time()
    
    # Save vector store
    logger.info(f"Saving vector store to {vector_store_path}")
    os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)
    rag_engine.save_vector_store(vector_store_path)
    
    # Log statistics
    stats = rag_engine.get_usage_stats()
    logger.info("Vector store generation completed successfully")
    logger.info(f"Time taken: {end_time - start_time:.2f} seconds")
    logger.info(f"Number of chunks: {stats['vector_store_size']}")
    logger.info(f"API calls made: {stats['api_calls']}")
    logger.info(f"Vector store size: {os.path.getsize(os.path.join(vector_store_path, 'vector_store.pkl.gz')) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    main() 