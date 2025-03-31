import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging
import os

class FrenchInstituteProcessor:
    def __init__(self, data_path: str):
        """
        Initialize the French Institute Processor with the path to data files
        
        Args:
            data_path: Path to the directory containing French formation institute data
        """
        self.data_path = data_path
        self.institutes_df = None
        self.nsf_df = None
        
    def load_data(self):
        """Load the French formation institutes data and NSF classification"""
        try:
            # Load formation institutes data
            csv_path = os.path.join(self.data_path, 'public_ofs.csv')
            self.institutes_df = pd.read_csv(csv_path, sep=';', encoding='utf-8')
            
            # Load NSF classification (Nomenclature des Spécialités de Formation)
            nsf_path = os.path.join(self.data_path, 'ListeNSF.xlsx')
            self.nsf_df = pd.read_excel(nsf_path)
            
            logging.info(f"Loaded {len(self.institutes_df)} French formation institutes")
            logging.info(f"Loaded {len(self.nsf_df)} NSF specialties")
            
            return True
            
        except Exception as e:
            logging.error(f"Error loading French institutes data: {str(e)}")
            return False
    
    def get_institutes(self) -> pd.DataFrame:
        """Return the institutes dataframe"""
        if self.institutes_df is None:
            self.load_data()
        return self.institutes_df
    
    def get_nsf_specialties(self) -> pd.DataFrame:
        """Return the NSF specialties dataframe"""
        if self.nsf_df is None:
            self.load_data()
        return self.nsf_df
    
    def map_nsf_to_onet(self) -> Dict[str, List[str]]:
        """
        Create a mapping between NSF codes (French) and O*NET SOC codes (US)
        
        This is a placeholder - actual mapping requires domain expertise and manual work
        """
        # Placeholder mapping - this would be populated with actual mappings
        nsf_to_onet_map = {}
        
        # Return empty mapping for now
        return nsf_to_onet_map
    
    def get_institutes_by_specialty(self, nsf_code: str) -> pd.DataFrame:
        """
        Get all institutes offering programs with a specific NSF specialty code
        
        Args:
            nsf_code: NSF specialty code to filter by
            
        Returns:
            DataFrame of institutes offering the specialty
        """
        if self.institutes_df is None:
            self.load_data()
            
        # Check all specialty columns (there are 15 possible specialty columns in the data)
        matching_institutes = pd.DataFrame()
        
        for i in range(1, 16):
            specialty_col = f'code_specialite_{i}'
            if specialty_col in self.institutes_df.columns:
                # Find institutes with this specialty
                specialty_match = self.institutes_df[self.institutes_df[specialty_col] == nsf_code]
                matching_institutes = pd.concat([matching_institutes, specialty_match])
        
        # Remove duplicates
        matching_institutes = matching_institutes.drop_duplicates()
        
        return matching_institutes
    
    def transform_to_neo4j_format(self) -> Dict[str, Any]:
        """
        Transform the French data into a format suitable for Neo4j import
        
        Returns:
            Dictionary with nodes and relationships for Neo4j
        """
        if self.institutes_df is None or self.nsf_df is None:
            self.load_data()
            
        # Create nodes for institutes
        institute_nodes = []
        for _, row in self.institutes_df.iterrows():
            institute = {
                'id': f"FR_INST_{row['siren']}_{row['num_etablissement']}",
                'name': row['raison_sociale'],
                'siren': row['siren'],
                'num_etablissement': row['num_etablissement'],
                'num_da': row['num_da'],
                'is_cfa': row['cfa'] == 'Oui',
                'address': f"{row['adresse_voie']}, {row['adresse_code_postal']} {row['adresse_ville']}",
                'postal_code': row['adresse_code_postal'],
                'city': row['adresse_ville'],
                'country': 'France'
            }
            institute_nodes.append(institute)
            
        # Create nodes for NSF specialties
        specialty_nodes = []
        for _, row in self.nsf_df.iterrows():
            # Assuming NSF_df has 'code' and 'libelle' columns
            specialty = {
                'id': f"FR_NSF_{row['code']}",
                'code': row['code'],
                'name': row['libelle'],
                'description': row.get('description', '')
            }
            specialty_nodes.append(specialty)
            
        # Create relationships between institutes and specialties
        institute_specialty_rels = []
        for _, row in self.institutes_df.iterrows():
            institute_id = f"FR_INST_{row['siren']}_{row['num_etablissement']}"
            
            # Check all specialty columns
            for i in range(1, 16):
                specialty_col = f'code_specialite_{i}'
                students_col = f'nb_stagiaires_{i}'
                hours_col = f'nb_heures_stagiaires_{i}'
                
                if specialty_col in row and pd.notna(row[specialty_col]) and row[specialty_col]:
                    specialty_id = f"FR_NSF_{row[specialty_col]}"
                    relationship = {
                        'from': institute_id,
                        'to': specialty_id,
                        'type': 'OFFERS_PROGRAM',
                        'properties': {
                            'num_students': int(row.get(students_col, 0)) if pd.notna(row.get(students_col, 0)) else 0,
                            'total_hours': int(row.get(hours_col, 0)) if pd.notna(row.get(hours_col, 0)) else 0
                        }
                    }
                    institute_specialty_rels.append(relationship)
        
        return {
            'institute_nodes': institute_nodes,
            'specialty_nodes': specialty_nodes,
            'institute_specialty_relationships': institute_specialty_rels
        }
    
    def create_education_path_nodes(self) -> List[Dict[str, Any]]:
        """
        Create education path nodes for the Neo4j graph
        
        Returns:
            List of education path nodes
        """
        if self.nsf_df is None:
            self.load_data()
            
        education_paths = []
        for _, row in self.nsf_df.iterrows():
            # Create an education path for each NSF specialty
            education_path = {
                'id': f"FR_EDUC_{row['code']}",
                'name': row['libelle'],
                'type': 'french_professional_training',
                'country': 'France',
                'nsf_code': row['code'],
                'description': row.get('description', '')
            }
            education_paths.append(education_path)
            
        return education_paths 