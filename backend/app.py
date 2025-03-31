import os
import logging
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from rag_engine import CareerGuidanceRAG

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("career_guidance_api")

app = FastAPI(
    title="Career Guidance AI",
    description="AI-powered career guidance using RAG",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Pydantic models
class QuestionRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class QuestionResponse(BaseModel):
    answer: str
    sources: List[str]

class ChatHistoryItem(BaseModel):
    question: str
    answer: str

class PersonalityAssessment(BaseModel):
    interests: List[str]
    skills: List[str]
    values: List[str]
    personality_traits: List[str]

# Global RAG engine
rag_engine = None

# Dependency to get the RAG engine
def get_rag_engine():
    global rag_engine
    if rag_engine is None:
        sql_dir = os.getenv("ONET_SQL_DIR", "./data/documentation/ONET")
        vector_store_path = os.getenv("VECTOR_STORE_PATH", "./vector_store")
        
        rag_engine = CareerGuidanceRAG(sql_dir)
        
        # Check if vector store exists
        if os.path.exists(vector_store_path):
            try:
                rag_engine.load_vector_store(vector_store_path)
                logger.info("Loaded existing vector store")
            except Exception as e:
                logger.error(f"Error loading vector store: {str(e)}")
                # If loading fails, create a new one
                rag_engine.load_data()
                rag_engine.save_vector_store(vector_store_path)
        else:
            # Create and save vector store
            rag_engine.load_data()
            os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)
            rag_engine.save_vector_store(vector_store_path)
            
        rag_engine.initialize_chat_engine()
    
    return rag_engine

@app.get("/")
def home():
    return {"message": "Career Guidance AI API is running"}

@app.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest, rag_engine: CareerGuidanceRAG = Depends(get_rag_engine)):
    """Ask a question to the career guidance system"""
    try:
        answer = rag_engine.ask(request.question)
        
        # Get sources from the last query results
        sources = []
        if hasattr(rag_engine, 'retriever') and rag_engine.retriever:
            try:
                # This is a simplification - actual source extraction depends on the retriever implementation
                docs = rag_engine.retriever.get_relevant_documents(request.question)
                sources = list(set([doc.metadata.get("source", "unknown") for doc in docs]))
            except:
                pass
        
        return QuestionResponse(
            answer=answer,
            sources=sources
        )
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/career-guidance")
def get_career_guidance(assessment: PersonalityAssessment, rag_engine: CareerGuidanceRAG = Depends(get_rag_engine)):
    """Get personalized career guidance based on assessment"""
    try:
        # Construct a natural language question from the assessment
        question = "Based on a personality assessment with "
        
        if assessment.interests:
            question += f"interests in {', '.join(assessment.interests)}, "
            
        if assessment.skills:
            question += f"skills in {', '.join(assessment.skills)}, "
            
        if assessment.values:
            question += f"values like {', '.join(assessment.values)}, "
            
        if assessment.personality_traits:
            question += f"and personality traits such as {', '.join(assessment.personality_traits)}, "
            
        question += "what careers would be most suitable and what education paths should be considered?"
        
        # Get guidance using the RAG engine
        guidance = rag_engine.ask(question)
        
        return {
            "career_guidance": guidance,
            "assessment": assessment.dict()
        }
    except Exception as e:
        logger.error(f"Error generating career guidance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 