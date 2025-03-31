from neo4j import GraphDatabase
from typing import Dict, List, Any
import logging

class Neo4jProcessor:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        """Close the Neo4j connection"""
        self.driver.close()
        
    # --- O*NET Data Methods ---
        
    def create_occupation(self, occupation_data: Dict[str, Any]) -> None:
        """Create an occupation node in Neo4j"""
        with self.driver.session() as session:
            query = """
            CREATE (o:Occupation {
                onetsoc_code: $onetsoc_code,
                title: $title,
                description: $description,
                alternate_titles: $alternate_titles
            })
            """
            session.run(query, **occupation_data)
            
    def create_skill(self, skill_data: Dict[str, Any]) -> None:
        """Create a skill node in Neo4j"""
        with self.driver.session() as session:
            query = """
            CREATE (s:Skill {
                element_id: $element_id,
                name: $name,
                description: $description,
                category: $category
            })
            """
            session.run(query, **skill_data)
            
    def create_occupation_skill_relationship(self, onetsoc_code: str, skill_data: Dict[str, Any]) -> None:
        """Create a relationship between occupation and skill"""
        with self.driver.session() as session:
            query = """
            MATCH (o:Occupation {onetsoc_code: $onetsoc_code})
            MATCH (s:Skill {element_id: $element_id})
            CREATE (o)-[r:REQUIRES {
                scale_id: $scale_id,
                data_value: $data_value
            }]->(s)
            """
            session.run(query, onetsoc_code=onetsoc_code, **skill_data)
            
    def create_education_requirement(self, onetsoc_code: str, education_data: Dict[str, Any]) -> None:
        """Create education requirement node and relationship"""
        with self.driver.session() as session:
            query = """
            MATCH (o:Occupation {onetsoc_code: $onetsoc_code})
            CREATE (e:EducationRequirement {
                education_level: $education_level,
                required_years: $required_years,
                description: $description
            })
            CREATE (o)-[:REQUIRES_EDUCATION]->(e)
            """
            session.run(query, onetsoc_code=onetsoc_code, **education_data)
            
    def update_occupation_with_bls_data(self, onetsoc_code: str, bls_data: Dict[str, Any]) -> None:
        """Update occupation node with BLS data"""
        with self.driver.session() as session:
            query = """
            MATCH (o:Occupation {onetsoc_code: $onetsoc_code})
            SET o += {
                employment_trend: $employment_trend,
                median_salary: $median_salary,
                growth_rate: $growth_rate,
                job_outlook: $job_outlook
            }
            """
            session.run(query, onetsoc_code=onetsoc_code, **bls_data)
            
    def create_industry_relationship(self, onetsoc_code: str, industry_data: Dict[str, Any]) -> None:
        """Create industry node and relationship with occupation"""
        with self.driver.session() as session:
            query = """
            MATCH (o:Occupation {onetsoc_code: $onetsoc_code})
            MERGE (i:Industry {code: $industry_code})
            SET i += {
                name: $industry_name,
                description: $industry_description,
                growth_rate: $growth_rate
            }
            CREATE (o)-[:BELONGS_TO]->(i)
            """
            session.run(query, onetsoc_code=onetsoc_code, **industry_data)
    
    # --- French Data Methods ---
    
    def create_french_institute(self, institute_data: Dict[str, Any]) -> None:
        """Create a French institute node in Neo4j"""
        with self.driver.session() as session:
            query = """
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
            """
            session.run(query, **institute_data)
    
    def create_french_specialty(self, specialty_data: Dict[str, Any]) -> None:
        """Create a French NSF specialty node in Neo4j"""
        with self.driver.session() as session:
            query = """
            CREATE (s:Specialty:FrenchNSF {
                id: $id,
                code: $code,
                name: $name,
                description: $description,
                country: "France"
            })
            """
            session.run(query, **specialty_data)
    
    def create_french_education_path(self, education_path_data: Dict[str, Any]) -> None:
        """Create a French education path node in Neo4j"""
        with self.driver.session() as session:
            query = """
            CREATE (e:EducationPath:FrenchEducation {
                id: $id,
                name: $name,
                type: $type,
                country: $country,
                nsf_code: $nsf_code
            })
            """
            session.run(query, **education_path_data)
    
    def create_nsf_onet_relationship(self, nsf_code: str, onetsoc_code: str) -> None:
        """Create a relationship between French NSF specialty and O*NET occupation"""
        with self.driver.session() as session:
            query = """
            MATCH (s:FrenchNSF {code: $nsf_code})
            MATCH (o:Occupation {onetsoc_code: $onetsoc_code})
            CREATE (s)-[:RELATES_TO {relationship_type: "international_equivalent"}]->(o)
            """
            session.run(query, nsf_code=nsf_code, onetsoc_code=onetsoc_code)
    
    def create_institute_specialty_relationship(self, relationship_data: Dict[str, Any]) -> None:
        """Create a relationship between a French institute and specialty"""
        with self.driver.session() as session:
            query = """
            MATCH (i:FrenchInstitute {id: $from})
            MATCH (s:FrenchNSF {id: $to})
            CREATE (i)-[:OFFERS_PROGRAM {
                num_students: $properties.num_students,
                total_hours: $properties.total_hours
            }]->(s)
            """
            session.run(query, **relationship_data)
    
    def connect_education_path_to_occupation(self, education_path_id: str, onetsoc_code: str, relationship_type: str = "leads_to") -> None:
        """Connect an education path to an occupation"""
        with self.driver.session() as session:
            query = """
            MATCH (e:EducationPath {id: $education_path_id})
            MATCH (o:Occupation {onetsoc_code: $onetsoc_code})
            CREATE (e)-[r:LEADS_TO {type: $relationship_type}]->(o)
            """
            session.run(query, education_path_id=education_path_id, onetsoc_code=onetsoc_code, relationship_type=relationship_type)
            
    # --- International Data Methods ---
    
    def create_country_node(self, country_data: Dict[str, Any]) -> None:
        """Create a country node in Neo4j"""
        with self.driver.session() as session:
            query = """
            CREATE (c:Country {
                code: $code,
                name: $name,
                region: $region,
                language: $language
            })
            """
            session.run(query, **country_data)
    
    def create_international_occupation_mapping(self, source_code: str, target_code: str, 
                                              source_system: str, target_system: str,
                                              similarity_score: float) -> None:
        """Create a mapping between international occupation classification systems"""
        with self.driver.session() as session:
            query = """
            MERGE (s:OccupationCode {code: $source_code, system: $source_system})
            MERGE (t:OccupationCode {code: $target_code, system: $target_system})
            CREATE (s)-[:MAPS_TO {similarity_score: $similarity_score}]->(t)
            """
            session.run(query, source_code=source_code, target_code=target_code,
                      source_system=source_system, target_system=target_system,
                      similarity_score=similarity_score)
            
    def get_education_paths_for_occupation(self, onetsoc_code: str, country: str = None) -> List[Dict[str, Any]]:
        """Get education paths leading to a specific occupation, optionally filtered by country"""
        with self.driver.session() as session:
            if country:
                query = """
                MATCH (e:EducationPath)-[:LEADS_TO]->(o:Occupation {onetsoc_code: $onetsoc_code})
                WHERE e.country = $country
                RETURN e
                """
                result = session.run(query, onetsoc_code=onetsoc_code, country=country)
            else:
                query = """
                MATCH (e:EducationPath)-[:LEADS_TO]->(o:Occupation {onetsoc_code: $onetsoc_code})
                RETURN e
                """
                result = session.run(query, onetsoc_code=onetsoc_code)
                
            return [dict(record["e"]) for record in result] 