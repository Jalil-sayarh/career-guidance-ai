import os
import re
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from dotenv import load_dotenv
import sqlite3
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("country_rag")

class CountryDataParser:
    """Parse country-specific career data into structured format for RAG"""
    
    def __init__(self, data_dir: str):
        """Initialize the parser with data directory"""
        self.data_dir = data_dir
    
    def parse_formation_institute_data(self, country: str) -> str:
        """Parse formation institute data from SQLite database"""
        try:
            # Connect to the SQLite database
            db_path = os.path.join(self.data_dir, country, "db", "french_data.db")
            logger.info(f"Connecting to database: {db_path}")
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            query = """
                SELECT 
                    numero_declaration_activite,
                    denomination,
                    siret_etablissement_declarant,
                    "adresse_de_l'organisme_de_formation",
                    certification_qualiopi_ou_equivalent
                FROM public_ofs
            """
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            if not rows:
                logger.warning(f"No formation institute data found for {country}")
                return []
            
            # Format the data into text chunks
            text_chunks = []
            for row in rows:
                chunk = f"Formation Institute:\nRegistration Number: {row[0]}\nName: {row[1]}\nSIRET: {row[2]}\nAddress: {row[3]}\nQuality Certification: {row[4]}"
                text_chunks.append(chunk)
            
            return text_chunks
            
        except Exception as e:
            logger.error(f"Error parsing formation institute data: {str(e)}")
            return []


class CountryCareerRAG:
    """RAG-based career guidance system for country-specific data"""
    
    def __init__(self, countries_dir: str, country_code: str, openai_api_key: Optional[str] = None):
        """Initialize the country-specific RAG engine"""
        self.countries_dir = countries_dir
        self.country_code = country_code.upper()
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.vector_store = None
        self.llm = None
        self.conversation = None
        self.memory = ConversationBufferMemory()
        self.text_chunks = []
        self.data_parser = CountryDataParser(countries_dir)
        
        # Country-specific settings
        self.country_settings = {
            "FR": {
                "name": "France",
                "language": "French",
                "data_paths": {
                    "formation_institutes": os.path.join(countries_dir, "france", "formation-institutes"),
                    "education": os.path.join(countries_dir, "france", "education"),
                    "job_market": os.path.join(countries_dir, "france", "job-market")
                },
                "vector_store_path": os.path.join(countries_dir, "vector_store", "france")
            },
            "MA": {
                "name": "Morocco",
                "language": "Arabic",
                "data_paths": {
                    "education": os.path.join(countries_dir, "morocco", "education"),
                    "job_market": os.path.join(countries_dir, "morocco", "job-market")
                },
                "vector_store_path": os.path.join(countries_dir, "vector_store", "morocco")
            }
        }
        
        if self.country_code not in self.country_settings:
            raise ValueError(f"Unsupported country code: {country_code}")
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
    
    def load_data(self):
        """Load and process data for a specific country."""
        try:
            # Load formation institute data
            text_chunks = self.data_parser.parse_formation_institute_data(self.country_settings[self.country_code]["name"].lower())
            if not text_chunks:
                self.logger.warning(f"No data found for country {self.country_code}")
                return
            
            # Store the text chunks for later use
            self.text_chunks = text_chunks
            self.logger.info(f"Loaded {len(text_chunks)} formation institute records")
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def initialize_chat_engine(self):
        """Initialize the chat engine with the appropriate settings."""
        logger.info("Initializing chat engine")
        
        llm = ChatOpenAI(
            model_name="gpt-4-turbo-preview",
            temperature=0.7
        )
        
        # Initialize memory
        self.memory = ConversationBufferMemory()
        
        # Initialize conversation chain with basic settings
        self.conversation = ConversationChain(
            llm=llm,
            memory=self.memory,
            verbose=False
        )
        
        logger.info("Chat engine initialized successfully")
    
    def ask(self, question: str) -> str:
        """Ask a country-specific question to the RAG engine"""
        logger.info(f"User question for {self.country_settings[self.country_code]['name']}: {question}")
        
        try:
            # Get relevant documents from ONET vector store
            if not self.vector_store:
                raise ValueError("Vector store not initialized. Please call load_data() first.")
                
            onet_documents = self.vector_store.similarity_search(question, k=5)
            onet_context = "\n\n".join([doc.page_content for doc in onet_documents])
            
            # Add French formation institute data if available
            french_context = ""
            if self.text_chunks:
                # Filter formation institutes based on the question
                relevant_institutes = []
                for chunk in self.text_chunks:
                    if any(keyword in chunk.lower() for keyword in question.lower().split()):
                        relevant_institutes.append(chunk)
                
                if relevant_institutes:
                    french_context = "\n\nRelevant Formation Institutes:\n" + "\n\n".join(relevant_institutes[:3])
            
            # Combine contexts
            full_context = f"ONET Career Information:\n{onet_context}"
            if french_context:
                full_context += f"\n\n{french_context}"
            
            # Add context to memory
            self.memory.save_context(
                {"input": f"Using this context for {self.country_code}: {full_context}\n\n{question}"}, 
                {"output": ""}
            )
            
            # Generate the answer using the conversation
            if not self.conversation:
                self.initialize_chat_engine()
                
            response = self.conversation.predict(input=question)
            return response
            
        except Exception as e:
            logger.error(f"Error with RAG approach: {str(e)}")
            
            # Fallback to direct LLM
            try:
                if not self.llm:
                    self.llm = ChatOpenAI(
                        openai_api_key=self.openai_api_key,
                        model_name="gpt-4",
                        temperature=0
                    )
                
                direct_prompt = f"""You are a career guidance AI assistant specializing in {self.country_settings[self.country_code]['language']} career paths.
                
Answer this question about careers in {self.country_settings[self.country_code]['name']} as best you can:

{question}"""
                
                response = self.llm.predict(direct_prompt)
                return response
                
            except Exception as e2:
                logger.error(f"Fallback failed: {str(e2)}")
                return f"I apologize, but I encountered an error while processing your question about careers in {self.country_settings[self.country_code]['name']}. Please try asking in a different way."
    
    def save_vector_store(self):
        """Save the vector store for future use"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Please call load_data() first.")
        
        save_path = self.country_settings[self.country_code]["vector_store_path"]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        logger.info(f"Saving vector store to {save_path}")
        self.vector_store.save_local(save_path)
    
    def load_vector_store(self):
        """Load a previously saved vector store"""
        load_path = self.country_settings[self.country_code]["vector_store_path"]
        logger.info(f"Loading vector store from {load_path}")
        
        embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        self.vector_store = FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)


def main():
    """Main function to demonstrate country-specific RAG engine"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Country-Specific Career Guidance RAG Engine")
    parser.add_argument("--country", required=True, choices=["FR", "MA"], help="Country code (FR or MA)")
    parser.add_argument("--countries-dir", required=True, help="Directory containing country-specific data")
    parser.add_argument("--question", help="Question to ask the RAG engine")
    
    args = parser.parse_args()
    
    rag_engine = CountryCareerRAG(args.countries_dir, args.country)
    
    try:
        rag_engine.load_vector_store()
    except Exception as e:
        print(f"Error loading vector store: {str(e)}")
        print("Creating new vector store...")
        rag_engine.load_data()
        rag_engine.save_vector_store()
    
    rag_engine.initialize_chat_engine()
    
    if args.question:
        answer = rag_engine.ask(args.question)
        print(f"Question: {args.question}")
        print(f"Answer: {answer}")
    else:
        # Interactive mode
        print(f"\n==== Career Guidance AI - {args.country} ====")
        print("Ask questions about careers, education, or type 'exit' to quit")
        print("Examples:")
        print("- What are the typical education paths in this country?")
        print("- What are the most in-demand careers?")
        print("- What are the salary ranges for different professions?")
        
        while True:
            question = input("\nYour question: ")
            if question.lower() in ['exit', 'quit', 'q']:
                break
                
            print("\nThinking...\n")
            answer = rag_engine.ask(question)
            print(f"Answer: {answer}")


if __name__ == "__main__":
    main() 