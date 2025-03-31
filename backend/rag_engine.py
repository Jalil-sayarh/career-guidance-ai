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
    
    def load_data(self):
        """Load data from SQL files into the RAG engine"""
        logger.info(f"Loading data from SQL files in {self.sql_directory}")
        
        # Get list of SQL files
        sql_files = []
        for file in os.listdir(self.sql_directory):
            if file.endswith('.sql'):
                sql_files.append(os.path.join(self.sql_directory, file))
        
        # Process each SQL file
        documents = []
        for sql_file in sql_files:
            text_content = SQLFileParser.extract_data_as_text(sql_file)
            documents.append({
                "content": text_content,
                "source": os.path.basename(sql_file)
            })
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        texts = []
        sources = []
        
        for doc in documents:
            chunks = text_splitter.split_text(doc["content"])
            texts.extend(chunks)
            sources.extend([doc["source"]] * len(chunks))
        
        # Create vector store
        embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        
        # Add metadata
        metadatas = [{"source": source} for source in sources]
        
        logger.info(f"Creating vector store with {len(texts)} text chunks")
        self.vector_store = FAISS.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas
        )
        
        logger.info("Vector store created successfully")
    
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
        logger.info(f"User question: {question}")
        
        try:
            # Try to get relevant documents using the vector store directly
            if not self.vector_store:
                raise ValueError("Vector store not initialized. Please call load_data() first.")
                
            documents = self.vector_store.similarity_search(question, k=5)
            context = "\n\n".join([doc.page_content for doc in documents])
            
            # Add context to memory
            self.memory.save_context({"input": f"Using this context: {context}\n\n{question}"}, 
                                     {"output": ""})
            
            # Generate the answer using the conversation
            if not self.conversation:
                self.initialize_chat_engine()
                
            response = self.conversation.predict(input=question)
            return response
            
        except Exception as e:
            logger.error(f"Error with RAG approach: {str(e)}")
            
            # Simplest possible fallback - direct question to LLM
            try:
                if not self.llm:
                    self.llm = ChatOpenAI(
                        openai_api_key=self.openai_api_key,
                        model_name="gpt-4",
                        temperature=0
                    )
                
                # Simple, direct approach without any complex chains
                direct_prompt = f"""You are a career guidance AI assistant. 
                
Answer this question about careers and occupations as best you can:

{question}"""
                
                response = self.llm.predict(direct_prompt)
                return response
                
            except Exception as e2:
                logger.error(f"Fallback failed: {str(e2)}")
                return f"I apologize, but I encountered an error while processing your question. Please try asking in a different way."
    
    def save_vector_store(self, path: str):
        """Save the vector store for future use"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Please call load_data() first.")
        
        logger.info(f"Saving vector store to {path}")
        self.vector_store.save_local(path)
    
    def load_vector_store(self, path: str):
        """Load a previously saved vector store"""
        logger.info(f"Loading vector store from {path}")
        embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        self.vector_store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


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