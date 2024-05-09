from neo4j_db_connection import get_db_session


def index_exists(session):
    result = session.run("""
        SHOW INDEXES YIELD name WHERE name = "index_e90b3ae4"
        RETURN COUNT(*) AS count
    """)
    return result.single()["count"] > 0


def save_results_to_neo4j(results):
    with get_db_session() as session:
        # First, ensure all nodes are created with unique identifiers and indexed by text
        for node in results['nodes']:
            session.run("""
                MERGE (a:Requirement {id: $id})
                ON CREATE SET a.description = $description, a.class = $class
                ON MATCH SET a.description = $description, a.class = $class
            """, {
                "id": node['id'],
                "description": node['properties']['text'],
                "class": node['properties']['predicted_class']
            })

        # Check if index exists before creation
        if not index_exists(session):
            # Create index for the Requirement label on the text property
            session.run("CREATE INDEX FOR (r:Requirement) ON (r.description)")

        # Create relationships based on class
        # Assuming you want to link Functional requirements to Non-functional ones
        session.run("""
            MATCH (f:Requirement), (nf:Requirement)
            WHERE f.class = 'Functional' AND nf.class = 'Non-functional'
            MERGE (f)-[:RELATED_TO]->(nf)
        """)
