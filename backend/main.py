import os
from fastapi import FastAPI, HTTPException
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Load API Key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Initialize LangChain Model
llm = ChatOpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY)

# Define a prompt template for career guidance
prompt_template = PromptTemplate.from_template(
    "The user is a high school student looking for career and education guidance. "
    "They have these interests: {interests}. "
    "They have these skills: {skills}. "
    "Based on this, suggest the best career paths and possible school programs."
)

@app.get("/")
def home():
    return {"message": "Career Guidance AI is running!"}

@app.post("/career-advice")
def get_career_advice(interests: str, skills: str):
    try:
        prompt = prompt_template.format(interests=interests, skills=skills)
        response = llm.invoke(prompt)
        return {"career_advice": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
