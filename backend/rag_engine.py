import os
import re
import logging
import json
import gzip
import pickle
from datetime import datetime
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rag_engine")

class SQLFileParser:
    """Parse SQL files into structured data for RAG"""
    
    @staticmethod
    def extract_table_name(sql_file_path: str) -> str:
        """Extract table name from the SQL file name"""
        filename = os.path.basename(sql_file_path)
        # Remove the numbering prefix and .sql extension
        table_name = re.sub(r'^\d+_', '', filename).replace('.sql', '')
        return table_name
    
    @staticmethod
    def extract_insert_statements(sql_content: str) -> List[str]:
        """Extract INSERT statements from SQL content"""
        # Simple pattern matching for INSERT statements
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
            
        # Extract column definitions
        col_pattern = r'`(.*?)`\s+(.*?)(?:,|\n|\))'
        matches = re.findall(col_pattern, create_statement)
        
        for match in matches:
            if len(match) >= 2:
                column_name, column_type = match[0], match[1]
                columns[column_name] = column_type.strip()
                
        return columns
    
    @staticmethod
    def extract_data_as_text(sql_file_path: str) -> str:
        """Extract data from SQL file as text for ingestion into RAG"""
        logger.info(f"Processing SQL file: {sql_file_path}")
        
        try:
            with open(sql_file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            table_name = SQLFileParser.extract_table_name(sql_file_path)
            create_statement = SQLFileParser.extract_create_table(content)
            columns = SQLFileParser.parse_create_table(create_statement)
            
            # Build a text description of the table
            table_desc = f"Table: {table_name}\n"
            table_desc += "Columns:\n"
            
            for col_name, col_type in columns.items():
                table_desc += f"- {col_name} ({col_type})\n"
                
            # Extract a sample of data (first few INSERT statements)
            inserts = SQLFileParser.extract_insert_statements(content)
            if inserts:
                sample_count = min(5, len(inserts))
                table_desc += f"\nSample data ({sample_count} rows):\n"
                
                for i in range(sample_count):
                    # Clean up the INSERT statement to show just the values
                    values = re.search(r'VALUES\s*\((.*?)\);', inserts[i], re.DOTALL)
                    if values:
                        table_desc += f"Row {i+1}: {values.group(1)}\n"
            
            return table_desc
            
        except Exception as e:
            logger.error(f"Error processing SQL file {sql_file_path}: {str(e)}")
            return f"Error processing {os.path.basename(sql_file_path)}: {str(e)}"


class CareerGuidanceRAG:
    """RAG-based career guidance system"""
    
    def __init__(self, sql_directory: str, openai_api_key: Optional[str] = None):
        """Initialize the RAG engine"""
        self.sql_directory = sql_directory
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.vector_store = None
        self.llm = None
        self.conversation = None
        self.memory = ConversationBufferMemory()
        
        # Cache settings
        self.query_embedding_cache = {}
        self.max_cache_size = 1000
        
        # Usage statistics
        self.usage_stats = {
            "queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "api_calls": 0,
            "vector_store_size": 0,
            "last_update": None
        }
        
        # Try to load existing vector store
        vector_store_path = os.getenv("VECTOR_STORE_PATH", "./vector_store")
        if os.path.exists(vector_store_path):
            try:
                self.load_vector_store(vector_store_path)
                logger.info("Loaded existing vector store")
            except Exception as e:
                logger.warning(f"Failed to load existing vector store: {str(e)}")
                logger.info("Will create new vector store if needed")
    
    def cleanup_cache(self):
        """Clean up old cache entries if cache is too large"""
        if len(self.query_embedding_cache) > self.max_cache_size:
            # Remove oldest entries
            oldest_keys = list(self.query_embedding_cache.keys())[:-self.max_cache_size]
            for key in oldest_keys:
                del self.query_embedding_cache[key]
            logger.info(f"Cleaned up {len(oldest_keys)} old cache entries")
    
    def load_data(self):
        """Load data from SQL files into the RAG engine"""
        if self.vector_store is not None:
            logger.info("Vector store already loaded, skipping data loading")
            return
            
        logger.info(f"Loading data from SQL files in {self.sql_directory}")
        
        # Get list of SQL files
        sql_files = []
        for file in os.listdir(self.sql_directory):
            if file.endswith('.sql'):
                sql_files.append(os.path.join(self.sql_directory, file))
        
        # Process each SQL file in parallel
        documents = []
        with ThreadPoolExecutor() as executor:
            futures = []
            for sql_file in sql_files:
                future = executor.submit(SQLFileParser.extract_data_as_text, sql_file)
                futures.append((future, os.path.basename(sql_file)))
            
            for future, source in futures:
                try:
                    text_content = future.result()
                    documents.append({
                        "content": text_content,
                        "source": source
                    })
                except Exception as e:
                    logger.error(f"Error processing {source}: {str(e)}")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Process chunks in parallel
        texts = []
        sources = []
        with ThreadPoolExecutor() as executor:
            futures = []
            for doc in documents:
                future = executor.submit(text_splitter.split_text, doc["content"])
                futures.append((future, doc["source"]))
            
            for future, source in futures:
                try:
                    chunks = future.result()
                    texts.extend(chunks)
                    sources.extend([source] * len(chunks))
                except Exception as e:
                    logger.error(f"Error splitting chunks for {source}: {str(e)}")
        
        # Create embeddings for all texts at once
        embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        metadatas = [{"source": source} for source in sources]
        
        # Create embeddings for all texts
        text_embeddings = embeddings.embed_documents(texts)
        self.usage_stats["api_calls"] += 1
        
        # Create vector store with all embeddings at once
        self.vector_store = FAISS.from_embeddings(
            text_embeddings=text_embeddings,
            texts=texts,
            metadatas=metadatas
        )
        
        logger.info(f"Created vector store with {len(texts)} text chunks")
        
        # Save the vector store
        vector_store_path = os.getenv("VECTOR_STORE_PATH", "./vector_store")
        os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)
        self.save_vector_store(vector_store_path)
        
        # Update usage stats
        self.usage_stats["vector_store_size"] = len(texts)
        self.usage_stats["last_update"] = datetime.now().isoformat()
    
    def initialize_chat_engine(self):
        """Initialize the chat engine with the retriever"""
        logger.info("Initializing chat engine")
        
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Please call load_data() first.")
        
        self.llm = ChatOpenAI(
            openai_api_key=self.openai_api_key,
            model_name="gpt-4",
            temperature=0
        )
        
        # Create a simple conversation chain
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
        """Ask a question to the RAG engine"""
        self.usage_stats["queries"] += 1
        
        try:
            # Check cache for query embedding
            if question in self.query_embedding_cache:
                self.usage_stats["cache_hits"] += 1
                query_embedding = self.query_embedding_cache[question]
            else:
                self.usage_stats["cache_misses"] += 1
                query_embedding = self.embeddings.embed_query(question)
                self.query_embedding_cache[question] = query_embedding
                self.cleanup_cache()
            
            # Get relevant documents
            documents = self.vector_store.similarity_search_with_score(
                question,
                k=5
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
    
    def save_vector_store(self, path: str):
        """Save vector store with compression and version info"""
        # Add version info
        version_info = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "num_chunks": self.usage_stats["vector_store_size"]
        }
        
        # Save version info
        with open(os.path.join(path, "version.json"), "w") as f:
            json.dump(version_info, f)
        
        # Compress and save vector store data
        data = {
            "index": self.vector_store.index,
            "docstore": self.vector_store.docstore,
            "index_to_docstore_id": self.vector_store.index_to_docstore_id
        }
        
        with gzip.open(os.path.join(path, "vector_store.pkl.gz"), "wb") as f:
            pickle.dump(data, f)
    
    def load_vector_store(self, path: str):
        """Load vector store with error recovery"""
        try:
            # Load version info
            with open(os.path.join(path, "version.json"), "r") as f:
                version_info = json.load(f)
                logger.info(f"Loading vector store version {version_info['version']}")
            
            # Load compressed vector store
            with gzip.open(os.path.join(path, "vector_store.pkl.gz"), "rb") as f:
                data = pickle.load(f)
            
            # Reconstruct vector store
            embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
            self.vector_store = FAISS(
                embedding_function=embeddings,
                index=data["index"],
                docstore=data["docstore"],
                index_to_docstore_id=data["index_to_docstore_id"]
            )
            
            # Update usage stats
            self.usage_stats["vector_store_size"] = version_info["num_chunks"]
            self.usage_stats["last_update"] = version_info["created_at"]
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            # Try to recover by creating new vector store
            self.load_data()
            self.save_vector_store(path)
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics"""
        return {
            **self.usage_stats,
            "cache_size": len(self.query_embedding_cache)
        }


def main():
    """Main function to demonstrate RAG engine"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Career Guidance RAG Engine")
    parser.add_argument("--sql-dir", required=True, help="Directory containing SQL files")
    parser.add_argument("--save-path", help="Path to save the vector store")
    parser.add_argument("--load-path", help="Path to load a saved vector store")
    parser.add_argument("--question", help="Question to ask the RAG engine")
    
    args = parser.parse_args()
    
    rag_engine = CareerGuidanceRAG(args.sql_dir)
    
    if args.load_path:
        rag_engine.load_vector_store(args.load_path)
    else:
        rag_engine.load_data()
        if args.save_path:
            rag_engine.save_vector_store(args.save_path)
    
    rag_engine.initialize_chat_engine()
    
    if args.question:
        answer = rag_engine.ask(args.question)
        print(f"Question: {args.question}")
        print(f"Answer: {answer}")
    else:
        # Interactive mode
        print("Career Guidance AI - Interactive Mode")
        print("Type 'exit' to quit")
        
        while True:
            question = input("\nYour question: ")
            if question.lower() == 'exit':
                break
                
            answer = rag_engine.ask(question)
            print(f"Answer: {answer}")


if __name__ == "__main__":
    main() 