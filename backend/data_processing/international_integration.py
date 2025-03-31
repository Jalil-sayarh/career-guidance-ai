import logging
from typing import Dict, List, Any
from neo4j import GraphDatabase
import pandas as pd
import os

from .onet_processor import ONETProcessor
from .neo4j_processor import Neo4jProcessor
from .french_institutes_processor import FrenchInstituteProcessor

class InternationalIntegration:
    def __init__(
        self,
        onet_config: Dict[str, str],
        neo4j_config: Dict[str, str],
        french_data_path: str
    ):
        """
        Initialize the international integration module
        
        Args:
            onet_config: Configuration for O*NET database connection
            neo4j_config: Configuration for Neo4j database connection
            french_data_path: Path to French data directory
        """
        self.onet_processor = ONETProcessor(**onet_config)
        self.neo4j_processor = Neo4jProcessor(**neo4j_config)
        self.french_processor = FrenchInstituteProcessor(french_data_path)
        
        # Load mapping tables - these would be maintained separately
        self.load_international_mappings()
        
    def load_international_mappings(self):
        """Load mapping tables for international data integration"""
        # These mappings would ideally be maintained in files or a database
        # For MVP purposes, we'll hardcode some example mappings
        
        # French NSF to O*NET SOC code mapping (sample)
        self.nsf_to_onet = {
            # IT related
            '326': ['15-1252.00', '15-1253.00', '15-1254.00'],  # Programming languages to Software Developers
            '326n': ['15-1211.00', '15-1212.00'],  # Network programming to Network Engineers
            '326t': ['15-1221.00', '15-1299.00'],  # Telecommunications to Telecom Engineers
            
            # Business related
            '310': ['13-1111.00', '13-1161.00', '13-1198.00'],  # Business management to Management Analysts
            '314': ['13-2011.00', '13-2051.00', '13-2099.00'],  # Finance to Financial Analysts
            
            # Engineering
            '200': ['17-2141.00', '17-2199.00'],  # Engineering technologies to Mechanical Engineers
            '201': ['17-2051.00', '17-2061.00'],  # Civil engineering to Civil Engineers
            
            # Healthcare
            '331': ['29-1221.00', '29-1228.00'],  # Health to Dentists
            '331p': ['29-1051.00', '29-1071.00'],  # Pharmacy to Pharmacists
            
            # Education
            '333': ['25-2021.00', '25-2022.00', '25-2031.00']  # Teaching to Teachers
        }
        
        # Education level mapping
        self.fr_education_to_us = {
            'CAP': 'High school diploma or equivalent',
            'BEP': 'High school diploma or equivalent',
            'Baccalauréat professionnel': 'Some college, no degree',
            'BTS': 'Associate\'s degree',
            'DUT': 'Associate\'s degree',
            'Licence': 'Bachelor\'s degree',
            'Master': 'Master\'s degree',
            'Doctorat': 'Doctoral degree'
        }
        
    def process_french_data(self):
        """
        Process French formation institute data and integrate with O*NET
        
        This function:
        1. Loads French formation institute data
        2. Creates nodes for French education institutions
        3. Creates nodes for French specialties (NSF codes)
        4. Maps French specialties to O*NET occupations
        5. Creates relationships in the Neo4j graph
        """
        # Load French data
        self.french_processor.load_data()
        
        # Transform to Neo4j format
        french_data = self.french_processor.transform_to_neo4j_format()
        
        # Create institute nodes
        for institute in french_data['institute_nodes']:
            self.neo4j_processor.create_french_institute(institute)
        
        # Create specialty nodes and education paths
        for specialty in french_data['specialty_nodes']:
            self.neo4j_processor.create_french_specialty(specialty)
            
            # Create education path for this specialty
            education_path = {
                'id': f"FR_EDUC_{specialty['code']}",
                'name': specialty['name'],
                'type': 'french_professional_training',
                'country': 'France',
                'nsf_code': specialty['code']
            }
            self.neo4j_processor.create_french_education_path(education_path)
            
            # Create relationships between specialty and O*NET occupations
            if specialty['code'] in self.nsf_to_onet:
                onet_codes = self.nsf_to_onet[specialty['code']]
                for onet_code in onet_codes:
                    self.neo4j_processor.create_nsf_onet_relationship(
                        nsf_code=specialty['code'],
                        onetsoc_code=onet_code
                    )
        
        # Create relationships between institutes and specialties
        for rel in french_data['institute_specialty_relationships']:
            self.neo4j_processor.create_institute_specialty_relationship(rel)
            
        logging.info(f"Processed {len(french_data['institute_nodes'])} French institutes")
        logging.info(f"Processed {len(french_data['specialty_nodes'])} French specialties")
        
    def get_french_career_paths(self, onetsoc_code: str) -> List[Dict[str, Any]]:
        """
        Get French career paths related to an O*NET occupation
        
        Args:
            onetsoc_code: O*NET SOC code
            
        Returns:
            List of French career paths
        """
        # This would query the Neo4j database to find mapped French specialties and institutes
        # Returning a placeholder for now
        return []
    
    def close(self):
        """Close all connections"""
        self.onet_processor.close()
        self.neo4j_processor.close() 