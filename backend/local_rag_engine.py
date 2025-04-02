import os
import re
import logging
import json
import gzip
import pickle
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from dotenv import load_dotenv
from difflib import SequenceMatcher
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("local_rag_engine")

# Common synonyms for career-related terms
CAREER_SYNONYMS = {
    "job": ["career", "occupation", "work", "profession"],
    "skill": ["ability", "competency", "expertise", "capability"],
    "education": ["training", "learning", "qualification", "degree"],
    "experience": ["background", "expertise", "knowledge", "practice"]
}

class SQLFileParser:
    """Parse SQL files into structured data for local search"""
    
    @staticmethod
    def extract_table_name(sql_file_path: str) -> str:
        """Extract table name from the SQL file name"""
        filename = os.path.basename(sql_file_path)
        table_name = re.sub(r'^\d+_', '', filename).replace('.sql', '')
        return table_name
    
    @staticmethod
    def extract_insert_statements(sql_content: str) -> List[str]:
        """Extract INSERT statements from SQL content"""
        insert_pattern = r'INSERT INTO.*?VALUES\s*\(.*?\);'
        inserts = re.findall(insert_pattern, sql_content, re.DOTALL | re.MULTILINE)
        return inserts
    
    @staticmethod
    def extract_create_table(sql_content: str) -> str:
        """Extract CREATE TABLE statement to understand schema"""
        create_pattern = r'CREATE TABLE.*?\);'
        matches = re.findall(create_pattern, sql_content, re.DOTALL | re.MULTILINE)
        if matches:
            return matches[0]
        return ""
    
    @staticmethod
    def parse_create_table(create_statement: str) -> Dict[str, str]:
        """Parse CREATE TABLE statement to get column information"""
        columns = {}
        if not create_statement:
            return columns
            
        col_pattern = r'`(.*?)`\s+(.*?)(?:,|\n|\))'
        matches = re.findall(col_pattern, create_statement)
        
        for match in matches:
            if len(match) >= 2:
                column_name, column_type = match[0], match[1]
                columns[column_name] = column_type.strip()
                
        return columns
    
    @staticmethod
    def extract_data_as_text(sql_file_path: str) -> Dict[str, Any]:
        """Extract data from SQL file as structured text"""
        logger.info(f"Processing SQL file: {sql_file_path}")
        
        try:
            with open(sql_file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            table_name = SQLFileParser.extract_table_name(sql_file_path)
            create_statement = SQLFileParser.extract_create_table(content)
            columns = SQLFileParser.parse_create_table(create_statement)
            
            # Extract all INSERT statements
            inserts = SQLFileParser.extract_insert_statements(content)
            
            # Process each INSERT statement
            rows = []
            for insert in inserts:
                values = re.search(r'VALUES\s*\((.*?)\);', insert, re.DOTALL)
                if values:
                    # Split values and clean them
                    row_values = [v.strip().strip("'") for v in values.group(1).split(',')]
                    # Create a dictionary mapping column names to values
                    row_dict = dict(zip(columns.keys(), row_values))
                    rows.append(row_dict)
            
            return {
                "table_name": table_name,
                "columns": columns,
                "rows": rows
            }
            
        except Exception as e:
            logger.error(f"Error processing SQL file {sql_file_path}: {str(e)}")
            return {"error": str(e)}

class LocalCareerGuidance:
    """Local-first career guidance system"""
    
    def __init__(self, sql_directory: str, openai_api_key: Optional[str] = None):
        """Initialize the local RAG engine"""
        self.sql_directory = sql_directory
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.data_cache = {}  # Cache for parsed SQL data
        self.vector_store = None  # Only created when needed
        self.llm = None
        self.conversation = None
        self.memory = ConversationBufferMemory()
        
        # Cache settings
        self.query_embedding_cache = {}
        self.max_cache_size = 1000
        self.cache_ttl = 3600  # 1 hour in seconds
        
        # Pagination settings
        self.page_size = 1000
        self.current_page = 0
        
        # Usage statistics
        self.usage_stats = {
            "queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "api_calls": 0,
            "local_matches": 0,
            "vector_store_size": 0,
            "last_update": None,
            "search_time": 0,
            "api_time": 0
        }
        
        # Error handling
        self.max_retries = 3
        self.retry_delay = 1  # seconds
    
    def load_data(self):
        """Load and parse SQL files into memory with pagination"""
        logger.info(f"Loading data from SQL files in {self.sql_directory}")
        
        # Get list of SQL files
        sql_files = []
        for file in os.listdir(self.sql_directory):
            if file.endswith('.sql'):
                sql_files.append(os.path.join(self.sql_directory, file))
        
        # Process files in batches
        total_files = len(sql_files)
        for i in range(0, total_files, self.page_size):
            batch_files = sql_files[i:i + self.page_size]
            
            # Process batch in parallel
            with ThreadPoolExecutor() as executor:
                futures = []
                for sql_file in batch_files:
                    future = executor.submit(SQLFileParser.extract_data_as_text, sql_file)
                    futures.append((future, os.path.basename(sql_file)))
                
                for future, source in futures:
                    try:
                        data = future.result()
                        if "error" not in data:
                            self.data_cache[source] = data
                    except Exception as e:
                        logger.error(f"Error processing {source}: {str(e)}")
            
            logger.info(f"Processed {min(i + self.page_size, total_files)}/{total_files} files")
        
        logger.info(f"Loaded {len(self.data_cache)} SQL files into memory")
    
    def get_synonyms(self, term: str) -> List[str]:
        """Get synonyms for a given term"""
        term = term.lower()
        synonyms = [term]
        
        for key, values in CAREER_SYNONYMS.items():
            if term == key or term in values:
                synonyms.extend([key] + values)
        
        return list(set(synonyms))
    
    def calculate_relevance_score(self, text: str, query_terms: List[str]) -> float:
        """Calculate relevance score for a text based on query terms"""
        text = text.lower()
        score = 0.0
        
        for term in query_terms:
            # Get synonyms for the term
            synonyms = self.get_synonyms(term)
            
            # Calculate best match among synonyms
            best_match = max(
                (SequenceMatcher(None, text, syn).ratio() for syn in synonyms),
                default=0.0
            )
            
            score += best_match
        
        return score / len(query_terms)
    
    def find_relevant_data(self, query: str) -> List[Tuple[Dict[str, Any], float]]:
        """Find relevant data using improved local search"""
        start_time = time.time()
        relevant_data = []
        query_terms = query.lower().split()
        
        for source, data in self.data_cache.items():
            # Calculate relevance scores for different parts
            table_score = self.calculate_relevance_score(data["table_name"], query_terms)
            columns_score = self.calculate_relevance_score(
                " ".join(data["columns"].keys()),
                query_terms
            )
            
            # Search in row values with scoring
            row_scores = []
            for row in data["rows"]:
                row_text = " ".join(str(v).lower() for v in row.values())
                score = self.calculate_relevance_score(row_text, query_terms)
                row_scores.append(score)
            
            # Calculate overall score
            max_row_score = max(row_scores) if row_scores else 0
            overall_score = max(table_score, columns_score, max_row_score)
            
            if overall_score > 0.3:  # Threshold for relevance
                relevant_data.append((data, overall_score))
        
        # Sort by relevance score
        relevant_data.sort(key=lambda x: x[1], reverse=True)
        
        # Update statistics
        self.usage_stats["local_matches"] += len(relevant_data)
        self.usage_stats["search_time"] += time.time() - start_time
        
        return relevant_data
    
    def create_embeddings_for_relevant_data(self, relevant_data: List[Tuple[Dict[str, Any], float]]) -> List[str]:
        """Create embeddings only for relevant data with relevance-based filtering"""
        texts = []
        
        # Sort by relevance score and take top results
        sorted_data = sorted(relevant_data, key=lambda x: x[1], reverse=True)
        top_data = sorted_data[:5]  # Only process top 5 most relevant results
        
        for data, score in top_data:
            # Create text representation of the data
            text = f"Table: {data['table_name']} (Relevance: {score:.2f})\n"
            text += "Columns:\n"
            for col_name, col_type in data["columns"].items():
                text += f"- {col_name} ({col_type})\n"
            
            # Add sample rows
            text += "\nSample data:\n"
            for row in data["rows"][:3]:  # Only include first 3 rows
                text += str(row) + "\n"
            
            texts.append(text)
        
        return texts
    
    def make_api_call_with_retry(self, func, *args, **kwargs) -> Any:
        """Make API call with retry mechanism"""
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                logger.warning(f"API call failed, attempt {attempt + 1}/{self.max_retries}: {str(e)}")
                time.sleep(self.retry_delay)
    
    def initialize_chat_engine(self):
        """Initialize the chat engine"""
        logger.info("Initializing chat engine")
        
        self.llm = ChatOpenAI(
            openai_api_key=self.openai_api_key,
            model_name="gpt-4",
            temperature=0
        )
        
        template = """
        You are a career guidance AI assistant with extensive knowledge about occupations, skills, and education paths.
        
        Use the following retrieved information to answer the user's question. If you don't know the answer based on the context provided, 
        just say that you don't know, don't try to make up an answer.
        
        {history}
        
        Human: {input}
        AI: """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        self.conversation = ConversationChain(
            llm=self.llm,
            prompt=prompt,
            memory=self.memory,
            verbose=False
        )
        
        logger.info("Chat engine initialized successfully")
    
    def ask(self, question: str) -> str:
        """Ask a question to the local RAG engine with improved error handling"""
        self.usage_stats["queries"] += 1
        start_time = time.time()
        
        try:
            # First, try to find relevant data using local search
            relevant_data = self.find_relevant_data(question)
            
            if not relevant_data:
                return "I couldn't find any relevant information in the database for your question."
            
            # Create embeddings only for relevant data
            texts = self.create_embeddings_for_relevant_data(relevant_data)
            
            # Create embeddings for the texts with retry mechanism
            embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
            text_embeddings = self.make_api_call_with_retry(
                embeddings.embed_documents,
                texts
            )
            self.usage_stats["api_calls"] += 1
            self.usage_stats["api_time"] += time.time() - start_time
            
            # Create a temporary vector store for similarity search
            self.vector_store = FAISS.from_embeddings(
                text_embeddings=text_embeddings,
                texts=texts
            )
            
            # Get most relevant documents
            documents = self.vector_store.similarity_search_with_score(
                question,
                k=3
            )
            
            # Format context
            context = "\n\n".join([doc.page_content for doc, _ in documents])
            
            # Generate response
            if not self.conversation:
                self.initialize_chat_engine()
            
            response = self.conversation.predict(input=question)
            return response
            
        except Exception as e:
            logger.error(f"Error in ask method: {str(e)}")
            return f"I apologize, but I encountered an error: {str(e)}"
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics"""
        return {
            **self.usage_stats,
            "cache_size": len(self.query_embedding_cache),
            "data_cache_size": len(self.data_cache),
            "average_search_time": self.usage_stats["search_time"] / max(1, self.usage_stats["queries"]),
            "average_api_time": self.usage_stats["api_time"] / max(1, self.usage_stats["api_calls"])
        }


def main():
    """Main function to demonstrate local RAG engine"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Local Career Guidance RAG Engine")
    parser.add_argument("--sql-dir", required=True, help="Directory containing SQL files")
    parser.add_argument("--question", help="Question to ask the RAG engine")
    
    args = parser.parse_args()
    
    rag_engine = LocalCareerGuidance(args.sql_dir)
    rag_engine.load_data()
    
    if args.question:
        answer = rag_engine.ask(args.question)
        print(f"Question: {args.question}")
        print(f"Answer: {answer}")
    else:
        # Interactive mode
        print("Local Career Guidance AI - Interactive Mode")
        print("Type 'exit' to quit")
        
        while True:
            question = input("\nYour question: ")
            if question.lower() == 'exit':
                break
                
            answer = rag_engine.ask(question)
            print(f"Answer: {answer}")


if __name__ == "__main__":
    main() 