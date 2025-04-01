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
logger = logging.getLogger("country_rag")

class CountryDataParser:
    """Parse country-specific career data into structured format for RAG"""
    
    @staticmethod
    def parse_formation_institute_data(file_path: str) -> str:
        """Parse French formation institute data from CSV or Excel files"""
        logger.info(f"Processing formation institute data: {file_path}")
        
        text_content = []
        
        try:
            # Read the file based on its extension and name
            if file_path.endswith('public_ofs_v2.xlsx'):
                df = pd.read_excel(file_path)
                # Convert the data into a readable text format
                text_content.append("Formation Institute Data:")
                text_content.append(f"Total institutes: {len(df)}")
                
                # Add summary of certifications
                cert_counts = df["Certification QUALIOPI ou équivalent"].value_counts()
                text_content.append("\nCertification Status:")
                for cert, count in cert_counts.items():
                    text_content.append(f"- {cert}: {count} institutes")
                
                # Add summary of training areas
                text_content.append("\nTraining Information:")
                text_content.append(f"Total trainers: {df['Effectif Formateurs'].sum()}")
                text_content.append(f"Total trainees: {df['Nb Stagiaires'].sum()}")
                
                # Add sample institutes
                text_content.append("\nSample Institutes:")
                for _, row in df.head().iterrows():
                    text_content.append(f"\nInstitute: {row['Dénomination']}")
                    text_content.append(f"Registration: {row['Numéro Déclaration Activité']}")
                    text_content.append(f"Address: {row.get('Adresse de l''organisme de formation', 'N/A')}")
                    text_content.append(f"Trainers: {row['Effectif Formateurs']}")
                    text_content.append(f"Trainees: {row['Nb Stagiaires']}")
                
            elif file_path.endswith('ListeNSF.xlsx'):
                df = pd.read_excel(file_path)
                # Convert the NSF (Nomenclature des Spécialités de Formation) data
                text_content.append("Training Specialties Classification (NSF):")
                
                # Skip empty rows and format the data
                for _, row in df.dropna().iterrows():
                    code = row['Unnamed: 0']
                    specialty = row['Unnamed: 1']
                    if pd.notna(code) and pd.notna(specialty):
                        text_content.append(f"- {int(code)}: {specialty}")
                
            elif file_path.endswith('Dessin_Enregistrement_ListeOF.xlsx'):
                df = pd.read_excel(file_path)
                # Convert the field descriptions
                text_content.append("Training Institute Data Fields Description:")
                
                for _, row in df.iterrows():
                    if pd.notna(row['Nom']) and pd.notna(row['Description']):
                        text_content.append(f"- {row['Nom']}: {row['Description']}")
                
            else:
                logger.warning(f"Skipping unsupported file: {file_path}")
                return f"Skipped file: {os.path.basename(file_path)}"
            
            return "\n".join(text_content)
            
        except Exception as e:
            logger.error(f"Error processing formation institute data {file_path}: {str(e)}")
            return f"Error processing {os.path.basename(file_path)}: {str(e)}"


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
    
    def load_data(self):
        """Load country-specific data into the RAG engine"""
        logger.info(f"Loading data for country: {self.country_settings[self.country_code]['name']}")
        
        country_setting = self.country_settings[self.country_code]
        documents = []
        
        # Process data based on country and available data paths
        for data_type, data_path in country_setting["data_paths"].items():
            if not os.path.exists(data_path):
                logger.warning(f"Data path not found: {data_path}")
                continue
                
            logger.info(f"Processing {data_type} data from {data_path}")
            
            if self.country_code == "FR":
                if data_type == "formation_institutes":
                    # Process French formation institute data
                    for file in os.listdir(data_path):
                        if file.endswith('.txt'):  # Adjust file extension as needed
                            file_path = os.path.join(data_path, file)
                            text_content = CountryDataParser.parse_formation_institute_data(file_path)
                            documents.append({
                                "content": text_content,
                                "source": file,
                                "country": self.country_code,
                                "data_type": data_type
                            })
                elif data_type in ["education", "job_market"]:
                    # Process education and job market data
                    for file in os.listdir(data_path):
                        if file.endswith('.txt'):  # Adjust file extension as needed
                            file_path = os.path.join(data_path, file)
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            documents.append({
                                "content": content,
                                "source": file,
                                "country": self.country_code,
                                "data_type": data_type
                            })
        
        if not documents:
            raise ValueError(f"No data found for country {self.country_code}")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        texts = []
        sources = []
        countries = []
        data_types = []
        
        for doc in documents:
            chunks = text_splitter.split_text(doc["content"])
            texts.extend(chunks)
            sources.extend([doc["source"]] * len(chunks))
            countries.extend([doc["country"]] * len(chunks))
            data_types.extend([doc["data_type"]] * len(chunks))
        
        # Create vector store
        embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        
        # Add metadata
        metadatas = [
            {
                "source": source,
                "country": country,
                "data_type": data_type
            }
            for source, country, data_type in zip(sources, countries, data_types)
        ]
        
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
        
        # Create a country-specific conversation chain
        template = f"""
        You are a career guidance AI assistant specializing in {self.country_settings[self.country_code]['language']} career paths and education.
        
        Use the following retrieved information to answer the user's question about careers in {self.country_settings[self.country_code]['name']}.
        If you don't know the answer based on the context provided, just say that you don't know, don't try to make up an answer.
        
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
        """Ask a country-specific question to the RAG engine"""
        logger.info(f"User question for {self.country_settings[self.country_code]['name']}: {question}")
        
        try:
            # Try to get relevant documents using the vector store directly
            if not self.vector_store:
                raise ValueError("Vector store not initialized. Please call load_data() first.")
                
            documents = self.vector_store.similarity_search(question, k=5)
            context = "\n\n".join([doc.page_content for doc in documents])
            
            # Add context to memory
            self.memory.save_context(
                {"input": f"Using this context for {self.country_code}: {context}\n\n{question}"}, 
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