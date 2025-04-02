import os
import re
import logging
import json
import gzip
import pickle
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("optimized_rag_engine")

# Common synonyms for career-related terms
CAREER_SYNONYMS = {
    "job": ["career", "occupation", "work", "profession"],
    "skill": ["ability", "competency", "expertise", "capability"],
    "education": ["training", "learning", "qualification", "degree"],
    "experience": ["background", "expertise", "knowledge", "practice"]
}

class OptimizedCareerGuidance:
    """Optimized local-first career guidance system"""
    
    def __init__(self, sql_dir: str):
        """Initialize the RAG engine with SQL directory path."""
        # Convert relative path to absolute path
        if not os.path.isabs(sql_dir):
            # Get the absolute path of the current file's directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Resolve the SQL directory path relative to the current directory
            sql_dir = os.path.abspath(os.path.join(current_dir, '..', sql_dir))
        
        # Ensure the SQL directory exists
        if not os.path.exists(sql_dir):
            raise FileNotFoundError(f"SQL directory not found: {sql_dir}")
        
        self.sql_dir = sql_dir
        self.logger = logging.getLogger(__name__)
        
        # Initialize OpenAI components
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=1000
        )
        self.embeddings = OpenAIEmbeddings()
        
        # Initialize memory
        self.memory = ConversationBufferMemory()
        
        # Initialize data structures
        self.data_cache = {}  # Cache for SQL data
        self.page_size = 1000  # Number of rows to process at once
        self.current_page = 0  # Current page number
        
        # Initialize usage stats
        self.usage_stats = {
            'total_queries': 0,
            'api_calls': 0,
            'local_matches': 0,
            'total_search_time': 0,
            'total_api_time': 0
        }
        
        # Cache settings
        self.query_embedding_cache = {}
        self.max_cache_size = 1000
        self.cache_ttl = 3600  # 1 hour in seconds
        
        # Error handling
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        
        # Load data
        self.load_data()
    
    def load_data(self):
        """Load and parse SQL files into memory with pagination"""
        logger.info(f"Loading data from SQL files in {self.sql_dir}")
        
        # Get list of SQL files
        sql_files = []
        for file in os.listdir(self.sql_dir):
            if file.endswith('.sql'):
                sql_files.append(os.path.join(self.sql_dir, file))
        
        # Process files in batches
        total_files = len(sql_files)
        for i in range(0, total_files, self.page_size):
            batch_files = sql_files[i:i + self.page_size]
            
            # Process batch in parallel
            with ThreadPoolExecutor() as executor:
                futures = []
                for sql_file in batch_files:
                    future = executor.submit(self._parse_sql_file, sql_file)
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
    
    def _parse_sql_file(self, sql_file_path: str) -> Dict[str, Any]:
        """Parse a single SQL file with improved handling of software development data."""
        try:
            logger.info(f"Parsing file: {sql_file_path}")
            table_name = os.path.basename(sql_file_path).replace('.sql', '')
            
            # Read file content
            with open(sql_file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Extract CREATE TABLE statement and columns
            create_pattern = r'CREATE TABLE[^;]+;'
            create_match = re.search(create_pattern, content, re.DOTALL)
            columns = {}
            
            if create_match:
                create_stmt = create_match.group(0)
                col_pattern = r'`([^`]+)`\s+([^,\n)]+)'
                for col_name, col_type in re.findall(col_pattern, create_stmt):
                    columns[col_name.strip()] = col_type.strip()
                logger.info(f"Found {len(columns)} columns in {table_name}")
            
            # Define software development related terms
            software_terms = [
                'software', 'programming', 'development', 'coding', 'computer',
                'web', 'database', 'application', 'system', 'design', 'testing',
                'engineering', 'analysis', 'algorithm', 'architecture', 'cloud',
                'security', 'network', 'interface', 'api', 'framework', 'library',
                'platform', 'server', 'client', 'frontend', 'backend', 'fullstack',
                'mobile', 'desktop', 'embedded', 'agile', 'scrum', 'devops',
                'java', 'python', 'javascript', 'html', 'css', 'sql', 'git'
            ]
            
            # Extract INSERT statements with improved regex
            rows = []
            insert_pattern = r"INSERT\s+INTO\s+[^(]+\s*\(([^)]+)\)\s*VALUES\s*\(([^;]+)\)"
            
            for match in re.finditer(insert_pattern, content, re.IGNORECASE | re.DOTALL):
                try:
                    cols = [c.strip('` \n\r\t') for c in match.group(1).split(',')]
                    values_str = match.group(2).strip()
                    
                    # Handle multiple value sets
                    value_sets = []
                    current_set = []
                    current = ''
                    in_quote = False
                    in_parentheses = 0
                    
                    for char in values_str:
                        if char == "'" and not current.endswith('\\'):
                            in_quote = not in_quote
                            current += char
                        elif char == '(' and not in_quote:
                            in_parentheses += 1
                            if in_parentheses == 1:
                                continue
                            current += char
                        elif char == ')' and not in_quote:
                            in_parentheses -= 1
                            if in_parentheses == 0:
                                if current:
                                    current_set.append(current.strip().strip("'"))
                                if current_set:
                                    value_sets.append(current_set)
                                current_set = []
                                current = ''
                            else:
                                current += char
                        elif char == ',' and not in_quote and in_parentheses == 0:
                            if current:
                                current_set.append(current.strip().strip("'"))
                            current = ''
                        else:
                            current += char
                    
                    if current:
                        current_set.append(current.strip().strip("'"))
                    if current_set:
                        value_sets.append(current_set)
                    
                    for vals in value_sets:
                        if len(cols) == len(vals):
                            row = dict(zip(cols, vals))
                            
                            # Filter relevant rows based on table type and content
                            row_text = " ".join(str(v).lower() for v in row.values())
                            
                            # Check if the row is relevant based on table type
                            is_relevant = False
                            
                            # Technology and skills tables
                            if any(term in table_name.lower() for term in ['technology', 'skills', 'knowledge', 'abilities']):
                                is_relevant = any(term in row_text for term in software_terms)
                            
                            # Work activities and tasks
                            elif any(term in table_name.lower() for term in ['work_activities', 'task', 'tools']):
                                is_relevant = any(term in row_text for term in software_terms)
                            
                            # Education and training
                            elif any(term in table_name.lower() for term in ['education', 'training', 'certification']):
                                is_relevant = any(term in row_text for term in software_terms)
                            
                            # Job titles and descriptions
                            elif any(term in table_name.lower() for term in ['occupation', 'job', 'title']):
                                is_relevant = any(term in row_text for term in software_terms)
                            
                            # Include row if relevant
                            if is_relevant:
                                # Add metadata to help with formatting
                                row['table_type'] = table_name
                                row['relevance_terms'] = [term for term in software_terms if term in row_text]
                                rows.append(row)
                            
                except Exception as e:
                    logger.error(f"Error processing row in {table_name}: {str(e)}")
                    continue
            
            logger.info(f"Found {len(rows)} relevant rows in {table_name}")
            return {
                "table_name": table_name,
                "columns": columns,
                "rows": rows[:500]  # Increased limit to 500 rows
            }
        
        except Exception as e:
            logger.error(f"Error parsing file {sql_file_path}: {str(e)}")
            return {"error": str(e)}
    
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
    
    def find_relevant_data(self, query: Union[str, List[str]]) -> List[Tuple[Dict[str, Any], float]]:
        """Find relevant data from the cache based on the query."""
        try:
            # Convert query to list of terms if it's a string
            if isinstance(query, str):
                query_terms = query.lower().split()
            else:
                query_terms = query
            
            relevant_data = []
            start_time = time.time()
            
            # Define relevant table patterns and their weights
            relevant_tables = {
                'technology_skills': 2.5,
                'skills': 2.0,
                'knowledge': 1.8,
                'work_activities': 1.5,
                'task': 1.3,
                'tools': 1.2,
                'work_styles': 1.2,
                'education_training': 1.1
            }
            
            # Define software development specific terms and their weights
            software_terms = {
                'software': 2.0,
                'programming': 2.0,
                'development': 1.8,
                'coding': 1.8,
                'computer': 1.5,
                'web': 1.5,
                'database': 1.5,
                'application': 1.4,
                'system': 1.3,
                'design': 1.3,
                'testing': 1.3,
                'engineering': 1.2,
                'analysis': 1.2
            }
            
            # Search through cached data
            for table_name, data in self.data_cache.items():
                if "rows" not in data:
                    continue
                
                # Calculate table relevance multiplier
                table_multiplier = 1.0
                for pattern, weight in relevant_tables.items():
                    if pattern in table_name.lower():
                        table_multiplier = weight
                        break
                
                for row in data["rows"]:
                    try:
                        # Convert row values to text for searching
                        row_text = " ".join(str(v).lower() for v in row.values())
                        
                        # Calculate base relevance score
                        base_score = self.calculate_relevance_score(row_text, query_terms)
                        
                        # Apply table multiplier
                        final_score = base_score * table_multiplier
                        
                        # Apply additional weights for software development terms
                        term_matches = 0
                        for term, weight in software_terms.items():
                            if term in row_text:
                                final_score *= weight
                                term_matches += 1
                        
                        # Boost score based on number of term matches
                        if term_matches > 0:
                            final_score *= (1 + (term_matches * 0.1))
                        
                        # Add context about the table name to help with formatting
                        if isinstance(row, dict):
                            row['table_name'] = table_name
                        
                        if final_score > 0.3:  # Threshold for relevance
                            relevant_data.append((row, final_score))
                    except Exception as e:
                        self.logger.error(f"Error processing row in {table_name}: {str(e)}")
                        continue
            
            # Sort by relevance score
            relevant_data.sort(key=lambda x: x[1], reverse=True)
            
            # Update usage stats
            self.usage_stats['local_matches'] += len(relevant_data)
            self.usage_stats['total_search_time'] += time.time() - start_time
            
            return relevant_data[:50]  # Return top 50 most relevant results
            
        except Exception as e:
            self.logger.error(f"Error in find_relevant_data: {str(e)}")
            return []
    
    def create_embeddings_for_relevant_data(self, relevant_data):
        """Format relevant data into structured categories with proper ONET data handling."""
        # Format data by category
        formatted_data = {
            'technical_skills': [],
            'core_skills': [],
            'knowledge': [],
            'work_activities': [],
            'tasks': [],
            'tools': []
        }
        
        # Process each entry in the relevant data
        for row, score in relevant_data:
            try:
                # Convert row to string for text matching
                row_text = str(row).lower()
                
                # Extract table name if available
                table_name = row.get('table_type', '').lower() if isinstance(row, dict) else ''
                
                # Determine category based on table name and content
                if any(term in table_name for term in ['technology', 'skills', 'abilities']):
                    if any(term in row_text for term in ['software', 'programming', 'coding', 'database']):
                        entry = self._format_tech_skill(row, score)
                        if entry:
                            formatted_data['technical_skills'].append(entry)
                    else:
                        entry = self._format_core_skill(row, score)
                        if entry:
                            formatted_data['core_skills'].append(entry)
                
                elif 'knowledge' in table_name or any(term in row_text for term in ['computer science', 'engineering', 'mathematics']):
                    entry = self._format_knowledge(row, score)
                    if entry:
                        formatted_data['knowledge'].append(entry)
                
                elif 'work_activities' in table_name or 'task_statements' in table_name:
                    if any(term in row_text for term in ['develop', 'design', 'implement', 'test']):
                        entry = self._format_work_activity(row, score)
                        if entry:
                            formatted_data['work_activities'].append(entry)
                    else:
                        entry = self._format_task(row, score)
                        if entry:
                            formatted_data['tasks'].append(entry)
                
                elif any(term in table_name for term in ['tools', 'technology']):
                    entry = self._format_tool(row, score)
                    if entry:
                        formatted_data['tools'].append(entry)
                
                # Handle occupation data
                elif 'occupation' in table_name:
                    if 'description' in row:
                        desc = row['description'].lower()
                        if any(term in desc for term in ['software', 'programming', 'development']):
                            # Extract skills from description
                            skills = [s.strip() for s in desc.split('.') if any(term in s for term in ['skill', 'ability', 'knowledge'])]
                            for skill in skills:
                                if any(term in skill for term in ['software', 'programming', 'coding', 'database']):
                                    formatted_data['technical_skills'].append(f"Technical Skill: {skill.capitalize()} (Score: {score:.2f})")
                                else:
                                    formatted_data['core_skills'].append(f"Core Skill: {skill.capitalize()} (Score: {score:.2f})")
            
            except Exception as e:
                self.logger.error(f"Error formatting row: {str(e)}")
                continue
        
        # Remove duplicates while preserving order
        for category in formatted_data:
            formatted_data[category] = list(dict.fromkeys(formatted_data[category]))
        
        # Create formatted text with headers
        formatted_text = """
SOFTWARE DEVELOPMENT SKILLS AND REQUIREMENTS
==========================================

This information is sourced from the O*NET database and organized by category.
Each entry includes relevance scores and detailed descriptions.

TECHNICAL SKILLS AND COMPETENCIES
-------------------------------
{technical_skills}

CORE PROFESSIONAL SKILLS
----------------------
{core_skills}

KNOWLEDGE AREAS
-------------
{knowledge}

COMMON WORK ACTIVITIES
-------------------
{work_activities}

TYPICAL TASKS
-----------
{tasks}

TOOLS AND TECHNOLOGIES
-------------------
{tools}

Note: Software development is a rapidly evolving field. Continuous learning and 
adaptation to new technologies and methodologies is essential for success.
""".format(
            technical_skills='\n'.join(formatted_data['technical_skills'][:5]) if formatted_data['technical_skills'] else 'No specific technical skills found.',
            core_skills='\n'.join(formatted_data['core_skills'][:5]) if formatted_data['core_skills'] else 'No specific core skills found.',
            knowledge='\n'.join(formatted_data['knowledge'][:5]) if formatted_data['knowledge'] else 'No specific knowledge areas found.',
            work_activities='\n'.join(formatted_data['work_activities'][:5]) if formatted_data['work_activities'] else 'No specific work activities found.',
            tasks='\n'.join(formatted_data['tasks'][:5]) if formatted_data['tasks'] else 'No specific tasks found.',
            tools='\n'.join(formatted_data['tools'][:5]) if formatted_data['tools'] else 'No specific tools found.'
        )
        
        return formatted_text

    def _format_tech_skill(self, row, score):
        """Format technical skill entry."""
        if isinstance(row, dict):
            description = None
            if 'title' in row:
                description = row['title']
            elif 'element_name' in row:
                description = row['element_name']
            elif 'scale_name' in row:
                description = row['scale_name']
            elif 'task_statement' in row:
                description = row['task_statement']
            elif 'description' in row:
                description = row['description']
            elif 'example' in row:
                description = row['example']
            
            if description:
                return f"Technical Skill: {description} (Score: {score:.2f})"
        return None

    def _format_core_skill(self, row, score):
        """Format core professional skill entry."""
        if isinstance(row, dict):
            description = None
            if 'element_name' in row:
                description = row['element_name']
            elif 'scale_name' in row:
                description = row['scale_name']
            elif 'description' in row:
                description = row['description']
            elif 'task_statement' in row:
                description = row['task_statement']
            
            if description:
                return f"Core Skill: {description} (Score: {score:.2f})"
        return None

    def _format_knowledge(self, row, score):
        """Format knowledge area entry."""
        if isinstance(row, dict):
            description = None
            if 'knowledge_type' in row:
                description = row['knowledge_type']
            elif 'element_name' in row:
                description = row['element_name']
            elif 'scale_name' in row:
                description = row['scale_name']
            elif 'description' in row:
                description = row['description']
            
            if description:
                return f"Knowledge Area: {description} (Score: {score:.2f})"
        return None

    def _format_work_activity(self, row, score):
        """Format work activity entry."""
        if isinstance(row, dict):
            description = None
            if 'activity' in row:
                description = row['activity']
            elif 'task_statement' in row:
                description = row['task_statement']
            elif 'element_name' in row:
                description = row['element_name']
            elif 'description' in row:
                description = row['description']
            
            if description:
                return f"Work Activity: {description} (Score: {score:.2f})"
        return None

    def _format_task(self, row, score):
        """Format task entry."""
        if isinstance(row, dict):
            description = None
            if 'task_statement' in row:
                description = row['task_statement']
            elif 'element_name' in row:
                description = row['element_name']
            elif 'description' in row:
                description = row['description']
            elif 'activity' in row:
                description = row['activity']
            
            if description:
                return f"Task: {description} (Score: {score:.2f})"
        return None

    def _format_tool(self, row, score):
        """Format tool/technology entry."""
        if isinstance(row, dict):
            description = None
            if 'example' in row:
                description = row['example']
            elif 'commodity_title' in row:
                description = row['commodity_title']
            elif 'element_name' in row:
                description = row['element_name']
            elif 'description' in row:
                description = row['description']
            
            if description:
                return f"Tool/Technology: {description} (Score: {score:.2f})"
        return None
    
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
    
    def ask(self, question: str) -> str:
        """Ask a question and get an answer using the RAG system."""
        try:
            start_time = time.time()
            self.usage_stats['total_queries'] += 1
            
            # Log the search
            self.logger.info(f"Searching for query: {question}")
            
            # Get query terms for local search
            query_terms = question.lower().split()
            self.logger.info(f"Query terms: {query_terms}")
            
            # Find relevant data using local search
            relevant_data = self.find_relevant_data(query_terms)
            
            # Create embeddings for the relevant data
            formatted_text = self.create_embeddings_for_relevant_data(relevant_data)
            
            # Create embeddings for the query
            query_embedding = self.embeddings.embed_query(question)
            
            # Create a temporary vector store for similarity search
            vector_store = FAISS.from_texts(
                [formatted_text],
                self.embeddings,
                metadatas=[{"source": "ONET Database"}]
            )
            
            # Get the most relevant documents
            docs = vector_store.similarity_search_with_score(question, k=3)
            
            # Format the context with relevance scores
            context = "Based on the ONET database:\n\n"
            for doc, score in docs:
                context += f"Relevance Score: {score:.2f}\n{doc.page_content}\n\n"
            
            # Create the prompt template
            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template="""You are a career guidance expert. Based on the following context from the ONET database, please provide a detailed answer to the question.

Context:
{context}

Question: {question}

Please provide a comprehensive answer that includes:
1. Key skills and competencies
2. Technical requirements
3. Related work activities
4. Important tools and technologies
5. Any relevant certifications or training

Answer:"""
            )
            
            # Create the LLM chain
            chain = LLMChain(
                llm=self.llm,
                prompt=prompt,
                verbose=True
            )
            
            # Get the answer
            response = chain.predict(context=context, question=question)
            
            # Update usage statistics
            self.usage_stats['api_calls'] += 1
            self.usage_stats['total_api_time'] += time.time() - start_time
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error in ask method: {str(e)}")
            return f"I apologize, but I encountered an error: {str(e)}"
    
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
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics"""
        return {
            **self.usage_stats,
            "cache_size": len(self.query_embedding_cache),
            "data_cache_size": len(self.data_cache),
            "average_search_time": self.usage_stats["total_search_time"] / max(1, self.usage_stats["total_queries"]),
            "average_api_time": self.usage_stats["total_api_time"] / max(1, self.usage_stats["api_calls"])
        }


def main():
    """Main function to demonstrate optimized RAG engine"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimized Career Guidance RAG Engine")
    parser.add_argument("--sql-dir", required=True, help="Directory containing SQL files")
    parser.add_argument("--question", help="Question to ask the RAG engine")
    
    args = parser.parse_args()
    
    # Get the absolute path of the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Resolve the SQL directory path relative to the current directory
    sql_dir = os.path.abspath(os.path.join(current_dir, '..', args.sql_dir))
    
    rag_engine = OptimizedCareerGuidance(sql_dir)
    rag_engine.load_data()
    
    if args.question:
        answer = rag_engine.ask(args.question)
        print(f"Question: {args.question}")
        print(f"Answer: {answer}")
        
        # Print usage statistics
        stats = rag_engine.get_usage_stats()
        print("\nUsage Statistics:")
        print(f"Total queries: {stats['total_queries']}")
        print(f"API calls: {stats['api_calls']}")
        print(f"Local matches: {stats['local_matches']}")
        print(f"Average search time: {stats['average_search_time']:.2f}s")
        print(f"Average API time: {stats['average_api_time']:.2f}s")
    else:
        # Interactive mode
        print("Optimized Career Guidance AI - Interactive Mode")
        print("Type 'exit' to quit")
        
        while True:
            question = input("\nYour question: ")
            if question.lower() == 'exit':
                break
                
            answer = rag_engine.ask(question)
            print(f"Answer: {answer}")
            
            # Print usage statistics after each question
            stats = rag_engine.get_usage_stats()
            print(f"\nAPI calls: {stats['api_calls']}, Local matches: {stats['local_matches']}")


if __name__ == "__main__":
    main() 