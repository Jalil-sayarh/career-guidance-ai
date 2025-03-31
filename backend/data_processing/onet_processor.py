import mysql.connector
from typing import Dict, List, Any
import logging

class ONETProcessor:
    def __init__(self, host: str, user: str, password: str, database: str):
        self.connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        self.cursor = self.connection.cursor(dictionary=True)
        
    def get_occupations(self) -> List[Dict[str, Any]]:
        """Fetch all occupations from O*NET database"""
        query = """
        SELECT 
            o.onetsoc_code,
            o.title,
            o.description,
            o.alternate_titles
        FROM occupation_data o
        """
        self.cursor.execute(query)
        return self.cursor.fetchall()
    
    def get_skills(self) -> List[Dict[str, Any]]:
        """Fetch all skills from O*NET database"""
        query = """
        SELECT 
            s.element_id,
            s.name,
            s.description,
            s.category
        FROM skills s
        """
        self.cursor.execute(query)
        return self.cursor.fetchall()
    
    def get_occupation_skills(self, onetsoc_code: str) -> List[Dict[str, Any]]:
        """Fetch skills required for a specific occupation"""
        query = """
        SELECT 
            s.element_id,
            s.name,
            s.description,
            os.scale_id,
            os.data_value
        FROM occupation_skills os
        JOIN skills s ON os.element_id = s.element_id
        WHERE os.onetsoc_code = %s
        """
        self.cursor.execute(query, (onetsoc_code,))
        return self.cursor.fetchall()
    
    def get_education_requirements(self, onetsoc_code: str) -> Dict[str, Any]:
        """Fetch education requirements for a specific occupation"""
        query = """
        SELECT 
            er.education_level,
            er.required_years,
            er.description
        FROM education_requirements er
        WHERE er.onetsoc_code = %s
        """
        self.cursor.execute(query, (onetsoc_code,))
        return self.cursor.fetchone()
    
    def close(self):
        """Close database connection"""
        self.cursor.close()
        self.connection.close() 