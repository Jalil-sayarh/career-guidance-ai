import os
import logging
import argparse
from dotenv import load_dotenv
from combined_rag_engine import CombinedCareerRAG
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("career_guidance")

# Initialize FastAPI app
app = FastAPI(title="Career Guidance AI API")

# Global RAG engine instance
rag_engine = None

class Question(BaseModel):
    text: str
    country: str = "FR"  # Default to France

def setup_environment():
    """Setup the environment and initialize the RAG engine"""
    global rag_engine
    
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    # Get directory paths from environment variables
    onet_sql_dir = os.getenv("ONET_SQL_DIR")
    countries_dir = os.getenv("COUNTRIES_DIR")
    
    if not onet_sql_dir or not countries_dir:
        raise ValueError("ONET_SQL_DIR and COUNTRIES_DIR environment variables must be set")
    
    # Initialize the combined RAG engine
    try:
        rag_engine = CombinedCareerRAG(
            onet_sql_dir=onet_sql_dir,
            countries_dir=countries_dir,
            country_code="FR"  # Default country
        )
        logger.info("Career Guidance RAG engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG engine: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    setup_environment()

@app.post("/ask")
async def ask_question(question: Question):
    """Handle a question about careers"""
    try:
        # Update country if different from default
        if question.country != rag_engine.country_rag.country_code:
            rag_engine.country_rag = CountryCareerRAG(
                rag_engine.country_rag.countries_dir,
                question.country,
                rag_engine.country_rag.openai_api_key
            )
            rag_engine.country_rag.load_data()
            rag_engine.country_rag.initialize_chat_engine()
        
        # Get response from RAG engine
        response = rag_engine.ask(question.text)
        return {"answer": response}
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def run_interactive_mode():
    """Run the application in interactive mode"""
    print("\n==== Career Guidance AI ====")
    print("Ask questions about careers, education, or type 'exit' to quit")
    print("Examples:")
    print("- What are the requirements for becoming a software developer?")
    print("- What are the typical education paths in this field?")
    print("- What are the salary ranges and job prospects?")
    
    while True:
        question = input("\nYour question: ")
        if question.lower() in ['exit', 'quit', 'q']:
            break
            
        print("\nThinking...\n")
        try:
            response = rag_engine.ask(question)
            print(f"Answer: {response}")
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            print(f"Error: {str(e)}")

def run_server():
    """Run the application as a FastAPI server"""
    uvicorn.run(app, host="0.0.0.0", port=8000)

def main():
    """Main function to run the application"""
    parser = argparse.ArgumentParser(description="Career Guidance AI")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--server", action="store_true", help="Run as a FastAPI server")
    
    args = parser.parse_args()
    
    if not (args.interactive or args.server):
        parser.error("Either --interactive or --server must be specified")
    
    try:
        setup_environment()
        
        if args.interactive:
            run_interactive_mode()
        elif args.server:
            run_server()
            
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 