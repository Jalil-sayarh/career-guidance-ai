import os
import logging
from typing import List, Dict, Any, Optional
from rag_engine import CareerGuidanceRAG
from country_rag_engine import CountryCareerRAG
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("combined_rag")


class CombinedCareerRAG:
    """Combined RAG engine that uses both ONET and country-specific data"""
    
    def __init__(self, onet_sql_dir: str, countries_dir: str, country_code: str, openai_api_key: Optional[str] = None):
        """Initialize the combined RAG engine"""
        self.onet_rag = CareerGuidanceRAG(onet_sql_dir, openai_api_key)
        self.country_rag = CountryCareerRAG(countries_dir, country_code, openai_api_key)
        
        # Initialize both engines
        self.onet_rag.load_data()
        self.country_rag.load_data()
        
        # Initialize chat engines
        self.onet_rag.initialize_chat_engine()
        self.country_rag.initialize_chat_engine()
    
    def ask(self, question: str) -> str:
        """Ask a question to the combined RAG engine"""
        logger.info(f"Processing question: {question}")
        
        try:
            # Get responses from both engines
            onet_response = self.onet_rag.ask(question)
            country_response = self.country_rag.ask(question)
            
            # Combine the responses
            combined_response = f"""Based on global career data (ONET):
{onet_response}

Based on country-specific data ({self.country_rag.country_settings[self.country_rag.country_code]['name']}):
{country_response}

Please consider both perspectives when making career decisions. The global data provides general information about the occupation, while the country-specific data gives you local context and requirements."""
            
            return combined_response
            
        except Exception as e:
            logger.error(f"Error in combined RAG: {str(e)}")
            
            # Fallback to individual responses if combined approach fails
            try:
                onet_response = self.onet_rag.ask(question)
                return f"Based on global career data:\n{onet_response}"
            except Exception as e2:
                logger.error(f"ONET RAG fallback failed: {str(e2)}")
                try:
                    country_response = self.country_rag.ask(question)
                    return f"Based on country-specific data:\n{country_response}"
                except Exception as e3:
                    logger.error(f"Country RAG fallback failed: {str(e3)}")
                    return "I apologize, but I encountered an error while processing your question. Please try asking in a different way."


def main():
    """Main function to demonstrate combined RAG engine"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Combined Career Guidance RAG Engine")
    parser.add_argument("--onet-dir", required=True, help="Directory containing ONET SQL files")
    parser.add_argument("--countries-dir", required=True, help="Directory containing country-specific data")
    parser.add_argument("--country", required=True, choices=["FR", "MA"], help="Country code (FR or MA)")
    parser.add_argument("--question", help="Question to ask the RAG engine")
    
    args = parser.parse_args()
    
    rag_engine = CombinedCareerRAG(args.onet_dir, args.countries_dir, args.country)
    
    if args.question:
        answer = rag_engine.ask(args.question)
        print(f"Question: {args.question}")
        print(f"Answer: {answer}")
    else:
        # Interactive mode
        print(f"\n==== Career Guidance AI - Global + {args.country} ====")
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
            answer = rag_engine.ask(question)
            print(f"Answer: {answer}")


if __name__ == "__main__":
    main() 