# Career Guidance AI - Data Processing Module

This module processes career data from multiple sources and integrates it into a Neo4j graph database for use in the Career Guidance AI application.

## Data Sources

1. **O*NET Database**: Comprehensive US-based occupational information
   - Skills, abilities, and knowledge requirements
   - Education and training paths
   - Work activities and context
   - Personality traits and work styles

2. **French Formation Institutes**: Education providers in France
   - Training programs by NSF specialty
   - Institution details and locations
   - Student enrollment and program hours

## Setup

### Prerequisites

- MySQL server with O*NET database loaded
- Neo4j server (4.0+)
- Python 3.8+
- Required Python packages (see requirements.txt)

### Environment Variables

Create a `.env` file with the following variables:

```
# O*NET Database
ONET_DB_HOST=localhost
ONET_DB_USER=your_mysql_user
ONET_DB_PASSWORD=your_mysql_password
ONET_DB_NAME=onet

# Neo4j Database
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password

# BLS API (optional)
BLS_API_KEY=your_bls_api_key

# French Data
FRENCH_DATA_PATH=./data/france-raw/formation-institutes
```

### Database Setup

1. **MySQL**: Import O*NET database
   ```bash
   # Import O*NET database files
   mysql -u your_mysql_user -p onet < 01_content_model_reference.sql
   mysql -u your_mysql_user -p onet < 02_job_zone_reference.sql
   # ... and so on for all SQL files
   ```

2. **Neo4j**: Install required plugins
   ```bash
   # For vectorization and semantic search
   neo4j-admin plugin install graphs-vectorize
   ```

## Running the Pipeline

### Process All Data

```bash
python -m backend.data_processing.main --all
```

### Process Specific Data Sources

```bash
# Process only O*NET data
python -m backend.data_processing.main --onet

# Process only French data
python -m backend.data_processing.main --french
```

## Graph Data Model

The graph database contains the following node types:

- **Occupation**: Career positions from O*NET
- **Skill**: Required skills for occupations
- **EducationPath**: Educational routes to careers
- **Industry**: Industry sectors
- **EducationInstitute**: Educational providers (US and international)
- **FrenchNSF**: French specialty classifications
- **Country**: Country-specific information

Key relationships include:

- `(Occupation)-[:REQUIRES]->(Skill)`
- `(Occupation)-[:REQUIRES_EDUCATION]->(EducationRequirement)`
- `(EducationPath)-[:LEADS_TO]->(Occupation)`
- `(EducationInstitute)-[:OFFERS_PROGRAM]->(FrenchNSF)`
- `(FrenchNSF)-[:RELATES_TO]->(Occupation)`

## International Data Integration

The system maps international occupation and education classifications to O*NET:

- French NSF codes → O*NET SOC codes
- French education levels → US education equivalents
- French training institutes → Neo4j graph structure

## Adding New Data Sources

To add a new international data source:

1. Create a processor class similar to `FrenchInstituteProcessor`
2. Implement mapping logic to O*NET classifications
3. Add Neo4j integration methods to `Neo4jProcessor`
4. Update `international_integration.py` to include the new source
5. Add processing function to `main.py` 