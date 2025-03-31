import os
import argparse
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("career_guidance")

def setup_environment():
    """Check if environment is properly set up"""
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set!")
        logger.info("Please set your OpenAI API key in the .env file")
        return False
    
    # Check for SQL directory
    sql_dir = os.getenv("ONET_SQL_DIR", "./data/documentation/ONET")
    if not os.path.exists(sql_dir):
        logger.error(f"SQL directory not found: {sql_dir}")
        logger.info("Please set the ONET_SQL_DIR environment variable to point to your SQL files")
        return False
    
    # Check for SQL files
    sql_files = [f for f in os.listdir(sql_dir) if f.endswith('.sql')]
    if not sql_files:
        logger.error(f"No SQL files found in {sql_dir}")
        return False
    
    logger.info(f"Found {len(sql_files)} SQL files in {sql_dir}")
    
    # Create vector store directory if it doesn't exist
    vector_store_path = os.getenv("VECTOR_STORE_PATH", "./vector_store")
    os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)
    
    return True

def run_interactive_mode():
    """Run the RAG engine in interactive mode"""
    from rag_engine import CareerGuidanceRAG
    
    # Initialize RAG engine
    sql_dir = os.getenv("ONET_SQL_DIR", "./data/documentation/ONET")
    vector_store_path = os.getenv("VECTOR_STORE_PATH", "./vector_store")
    
    print("\nInitializing Career Guidance RAG engine...")
    rag_engine = CareerGuidanceRAG(sql_dir)
    
    # Check if vector store exists
    if os.path.exists(vector_store_path):
        try:
            print("Loading existing vector store...")
            rag_engine.load_vector_store(vector_store_path)
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            print("Creating new vector store...")
            rag_engine.load_data()
            rag_engine.save_vector_store(vector_store_path)
    else:
        print("Creating new vector store...")
        rag_engine.load_data()
        os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)
        rag_engine.save_vector_store(vector_store_path)
    
    rag_engine.initialize_chat_engine()
    
    # Interactive mode
    print("\n==== Career Guidance AI ====")
    print("Ask questions about careers, skills, education, or type 'exit' to quit")
    print("Examples:")
    print("- What skills are needed for software developers?")
    print("- What education is required for nursing?")
    print("- What careers are good for people who like helping others?")
    print("- Suggest careers for someone with interests in technology and creativity")
    
    while True:
        question = input("\nYour question: ")
        if question.lower() in ['exit', 'quit', 'q']:
            break
            
        print("\nThinking...\n")
        answer = rag_engine.ask(question)
        print(f"Answer: {answer}")

def run_server():
    """Run the FastAPI server"""
    import uvicorn
    
    print("Starting Career Guidance API server...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

def main():
    parser = argparse.ArgumentParser(description="Career Guidance AI")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--server", action="store_true", help="Run as API server")
    
    args = parser.parse_args()
    
    if not setup_environment():
        return
    
    if args.interactive:
        run_interactive_mode()
    elif args.server:
        run_server()
    else:
        # Default to interactive mode
        run_interactive_mode()

if __name__ == "__main__":
    main() 