# Career Guidance AI - RAG Approach

This project implements a career guidance system using RAG (Retrieval-Augmented Generation) that directly operates on SQL files from the O*NET database without requiring a database setup.

## Features

- **Natural Language Interface**: Ask career-related questions in plain English
- **Direct SQL File Processing**: No database setup required
- **Personality Assessment**: Get career recommendations based on interests, skills, and personality traits
- **API Interface**: REST API for integration with frontend applications
- **Vector Storage**: Efficient storage and retrieval of career data

## Setup

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

1. Clone the repository
```bash
git clone https://github.com/your-username/career-guidance-ai.git
cd career-guidance-ai/backend
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Create a `.env` file from the template
```bash
cp .env.template .env
```

4. Edit the `.env` file with your OpenAI API key and data paths
```bash
OPENAI_API_KEY=your_openai_api_key_here
ONET_SQL_DIR=./data/documentation/ONET
VECTOR_STORE_PATH=./vector_store
```

### Running the Application

#### Interactive Mode

Run the application in interactive mode to chat with the career guidance system:

```bash
python run.py --interactive
```

#### API Server

Run the application as an API server:

```bash
python run.py --server
```

The API will be available at http://localhost:8000

API documentation is available at http://localhost:8000/docs

## API Endpoints

### `/ask` (POST)

Ask a question to the career guidance system.

Example request:
```json
{
  "question": "What skills are needed for software developers?"
}
```

Example response:
```json
{
  "answer": "Software developers typically need the following skills:\n1. Programming languages...",
  "sources": ["16_skills.sql", "03_occupation_data.sql"]
}
```

### `/career-guidance` (POST)

Get personalized career guidance based on assessment data.

Example request:
```json
{
  "interests": ["technology", "problem solving"],
  "skills": ["programming", "mathematics"],
  "values": ["creativity", "independence"],
  "personality_traits": ["analytical", "detail-oriented"]
}
```

Example response:
```json
{
  "career_guidance": "Based on your interests in technology and problem solving...",
  "assessment": {
    "interests": ["technology", "problem solving"],
    "skills": ["programming", "mathematics"],
    "values": ["creativity", "independence"],
    "personality_traits": ["analytical", "detail-oriented"]
  }
}
```

## How It Works

1. **SQL File Processing**: The system reads SQL files containing O*NET occupational data
2. **Text Extraction**: Relevant information is extracted from the SQL files
3. **Vectorization**: The text is converted into embeddings using OpenAI's embedding model
4. **Vector Storage**: Embeddings are stored in a FAISS vector database
5. **Retrieval**: When a question is asked, the system retrieves the most relevant chunks of information
6. **Generation**: GPT-4 generates an answer based on the retrieved information and the question

## Files

- `rag_engine.py`: Core implementation of the RAG engine
- `app.py`: FastAPI application for the REST API
- `run.py`: Script to run the application in interactive or server mode 