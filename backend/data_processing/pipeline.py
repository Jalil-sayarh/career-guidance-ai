import asyncio
from typing import Dict, List, Any
import logging
from .onet_processor import ONETProcessor
from .bls_processor import BLSProcessor
from .neo4j_processor import Neo4jProcessor

class DataPipeline:
    def __init__(
        self,
        onet_config: Dict[str, str],
        bls_api_key: str,
        neo4j_config: Dict[str, str]
    ):
        self.onet_processor = ONETProcessor(**onet_config)
        self.bls_processor = BLSProcessor(bls_api_key)
        self.neo4j_processor = Neo4jProcessor(**neo4j_config)
        
    async def process_occupation(self, onetsoc_code: str) -> None:
        """Process a single occupation and its related data"""
        try:
            # Get O*NET data
            occupation_data = self.onet_processor.get_occupations()
            skills_data = self.onet_processor.get_occupation_skills(onetsoc_code)
            education_data = self.onet_processor.get_education_requirements(onetsoc_code)
            
            # Get BLS data
            bls_data = await self.bls_processor.get_occupation_data(onetsoc_code)
            outlook_data = await self.bls_processor.get_occupation_outlook(onetsoc_code)
            salary_data = await self.bls_processor.get_salary_data(onetsoc_code)
            
            # Create nodes and relationships in Neo4j
            self.neo4j_processor.create_occupation(occupation_data)
            
            for skill in skills_data:
                self.neo4j_processor.create_skill(skill)
                self.neo4j_processor.create_occupation_skill_relationship(onetsoc_code, skill)
                
            if education_data:
                self.neo4j_processor.create_education_requirement(onetsoc_code, education_data)
                
            # Update with BLS data
            combined_bls_data = {
                **bls_data,
                **outlook_data,
                **salary_data
            }
            self.neo4j_processor.update_occupation_with_bls_data(onetsoc_code, combined_bls_data)
            
        except Exception as e:
            logging.error(f"Error processing occupation {onetsoc_code}: {str(e)}")
            raise
            
    async def process_all_occupations(self) -> None:
        """Process all occupations from O*NET database"""
        occupations = self.onet_processor.get_occupations()
        tasks = [self.process_occupation(occ['onetsoc_code']) for occ in occupations]
        await asyncio.gather(*tasks)
        
    def close(self):
        """Close all connections"""
        self.onet_processor.close()
        self.neo4j_processor.close() 