import os
import json
import logging
import argparse
from typing import Dict, List, Any
from neo4j import GraphDatabase

logger = logging.getLogger("test_data")

class TestDataGenerator:
    """Generate test data for validating the Career Guidance data pipeline"""
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        self.driver = GraphDatabase.driver(
            neo4j_uri, 
            auth=(neo4j_user, neo4j_password)
        )
    
    def generate_test_occupation(self) -> Dict[str, Any]:
        """Generate a test occupation"""
        return {
            "onetsoc_code": "TEST-00000.00",
            "title": "Test Occupation",
            "description": "This is a test occupation for validating the data pipeline.",
            "alternate_titles": "Sample Job; Example Position"
        }
    
    def generate_test_skills(self) -> List[Dict[str, Any]]:
        """Generate test skills"""
        return [
            {
                "element_id": "TEST-SKILL-1",
                "name": "Test Skill 1",
                "description": "This is test skill 1",
                "category": "Technical"
            },
            {
                "element_id": "TEST-SKILL-2",
                "name": "Test Skill 2",
                "description": "This is test skill 2",
                "category": "Social"
            }
        ]
    
    def generate_test_education(self) -> Dict[str, Any]:
        """Generate test education requirement"""
        return {
            "education_level": "Bachelor's degree",
            "required_years": 4,
            "description": "A bachelor's degree is typically required"
        }
    
    def generate_test_french_institute(self) -> Dict[str, Any]:
        """Generate test French institute"""
        return {
            "id": "FR_INST_TEST_001",
            "name": "Test French Institute",
            "siren": "TEST123456",
            "num_etablissement": "00001",
            "num_da": "TEST12345",
            "is_cfa": False,
            "address": "123 Test Street, 75001 Paris",
            "postal_code": "75001",
            "city": "Paris",
            "country": "France"
        }
    
    def generate_test_french_specialty(self) -> Dict[str, Any]:
        """Generate test French NSF specialty"""
        return {
            "id": "FR_NSF_TEST001",
            "code": "TEST001",
            "name": "Test French Specialty",
            "description": "This is a test French specialty",
            "country": "France"
        }
    
    def create_test_data_in_neo4j(self):
        """Create test data in Neo4j database"""
        with self.driver.session() as session:
            # First, check if test data already exists
            result = session.run("MATCH (o:Occupation) WHERE o.onetsoc_code = 'TEST-00000.00' RETURN COUNT(o) as count")
            if result.single()["count"] > 0:
                logger.info("Test data already exists in Neo4j")
                return
            
            # Create test occupation
            occupation = self.generate_test_occupation()
            session.run("""
                CREATE (o:Occupation {
                    onetsoc_code: $onetsoc_code,
                    title: $title,
                    description: $description,
                    alternate_titles: $alternate_titles
                })
            """, **occupation)
            
            # Create test skills
            skills = self.generate_test_skills()
            for skill in skills:
                session.run("""
                    CREATE (s:Skill {
                        element_id: $element_id,
                        name: $name,
                        description: $description,
                        category: $category
                    })
                """, **skill)
                
                # Create relationship between occupation and skill
                session.run("""
                    MATCH (o:Occupation {onetsoc_code: $onetsoc_code})
                    MATCH (s:Skill {element_id: $element_id})
                    CREATE (o)-[:REQUIRES {scale_id: "Test", data_value: 5.0}]->(s)
                """, onetsoc_code=occupation["onetsoc_code"], element_id=skill["element_id"])
            
            # Create education requirement
            education = self.generate_test_education()
            session.run("""
                MATCH (o:Occupation {onetsoc_code: $onetsoc_code})
                CREATE (e:EducationRequirement {
                    education_level: $education_level,
                    required_years: $required_years,
                    description: $description
                })
                CREATE (o)-[:REQUIRES_EDUCATION]->(e)
            """, onetsoc_code=occupation["onetsoc_code"], **education)
            
            # Create French institute
            institute = self.generate_test_french_institute()
            session.run("""
                CREATE (i:EducationInstitute:FrenchInstitute {
                    id: $id,
                    name: $name,
                    siren: $siren,
                    num_etablissement: $num_etablissement,
                    num_da: $num_da,
                    is_cfa: $is_cfa,
                    address: $address,
                    postal_code: $postal_code,
                    city: $city,
                    country: $country
                })
            """, **institute)
            
            # Create French specialty
            specialty = self.generate_test_french_specialty()
            session.run("""
                CREATE (s:Specialty:FrenchNSF {
                    id: $id,
                    code: $code,
                    name: $name,
                    description: $description,
                    country: $country
                })
            """, **specialty)
            
            # Create relationship between specialty and occupation
            session.run("""
                MATCH (s:FrenchNSF {code: $code})
                MATCH (o:Occupation {onetsoc_code: $onetsoc_code})
                CREATE (s)-[:RELATES_TO {relationship_type: "international_equivalent"}]->(o)
            """, code=specialty["code"], onetsoc_code=occupation["onetsoc_code"])
            
            # Create relationship between institute and specialty
            session.run("""
                MATCH (i:FrenchInstitute {id: $institute_id})
                MATCH (s:FrenchNSF {id: $specialty_id})
                CREATE (i)-[:OFFERS_PROGRAM {
                    num_students: 10,
                    total_hours: 100
                }]->(s)
            """, institute_id=institute["id"], specialty_id=specialty["id"])
            
            logger.info("Successfully created test data in Neo4j")
    
    def validate_test_data(self) -> bool:
        """Validate that test data can be retrieved correctly"""
        with self.driver.session() as session:
            # Test query 1: Get occupation with skills
            result = session.run("""
                MATCH (o:Occupation {onetsoc_code: 'TEST-00000.00'})-[:REQUIRES]->(s:Skill)
                RETURN o.title as occupation, collect(s.name) as skills
            """)
            
            record = result.single()
            if not record or record["occupation"] != "Test Occupation" or len(record["skills"]) != 2:
                logger.error("Failed to validate occupation-skill relationship")
                return False
                
            # Test query 2: Get French institutes for occupation
            result = session.run("""
                MATCH (i:FrenchInstitute)-[:OFFERS_PROGRAM]->(s:FrenchNSF)-[:RELATES_TO]->(o:Occupation {onetsoc_code: 'TEST-00000.00'})
                RETURN i.name as institute, s.name as specialty
            """)
            
            record = result.single()
            if not record or record["institute"] != "Test French Institute":
                logger.error("Failed to validate French institute-specialty-occupation path")
                return False
            
            logger.info("Successfully validated test data")
            return True
    
    def clean_test_data(self):
        """Remove test data from Neo4j"""
        with self.driver.session() as session:
            session.run("""
                MATCH (o:Occupation {onetsoc_code: 'TEST-00000.00'})
                OPTIONAL MATCH (o)-[r1]-()
                DELETE r1, o
            """)
            
            session.run("""
                MATCH (s:Skill) WHERE s.element_id STARTS WITH 'TEST-SKILL'
                OPTIONAL MATCH (s)-[r2]-()
                DELETE r2, s
            """)
            
            session.run("""
                MATCH (e:EducationRequirement {education_level: 'Bachelor\\'s degree', required_years: 4})
                OPTIONAL MATCH (e)-[r3]-()
                DELETE r3, e
            """)
            
            session.run("""
                MATCH (i:FrenchInstitute {id: 'FR_INST_TEST_001'})
                OPTIONAL MATCH (i)-[r4]-()
                DELETE r4, i
            """)
            
            session.run("""
                MATCH (s:FrenchNSF {code: 'TEST001'})
                OPTIONAL MATCH (s)-[r5]-()
                DELETE r5, s
            """)
            
            logger.info("Successfully cleaned test data")
    
    def close(self):
        """Close connection"""
        self.driver.close()

def main():
    parser = argparse.ArgumentParser(description="Test data tools for Career Guidance")
    parser.add_argument("--create", action="store_true", help="Create test data in Neo4j")
    parser.add_argument("--validate", action="store_true", help="Validate test data")
    parser.add_argument("--clean", action="store_true", help="Clean test data")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Load connection details from environment
    from dotenv import load_dotenv
    load_dotenv()
    
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "neo4j")
    
    generator = TestDataGenerator(neo4j_uri, neo4j_user, neo4j_password)
    
    try:
        if args.create:
            generator.create_test_data_in_neo4j()
        
        if args.validate:
            success = generator.validate_test_data()
            if success:
                logger.info("All test data validations passed!")
            else:
                logger.error("Test data validation failed!")
        
        if args.clean:
            generator.clean_test_data()
            
        if not (args.create or args.validate or args.clean):
            # Default behavior if no arguments provided
            generator.create_test_data_in_neo4j()
            success = generator.validate_test_data()
            if success:
                logger.info("All test data validations passed!")
            else:
                logger.error("Test data validation failed!")
    
    finally:
        generator.close()

if __name__ == "__main__":
    main() 