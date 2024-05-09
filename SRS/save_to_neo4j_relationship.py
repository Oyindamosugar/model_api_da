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


def save_results_to_neo4j_relationship(nodes_data, relationships_data):
    with get_db_session() as session:
        # First, ensure all nodes are created with unique identifiers and indexed by text
        for node_data in nodes_data:
            session.run("""
                MERGE (a:Requirements {id: $id})
                ON CREATE SET a.description = $description, a.class = $class
                ON MATCH SET a.description = $description, a.class = $class
            """, {
                "id": node_data['id'],
                "description": node_data['description'],
                "class": node_data['class']
            })

        # Create relationships based on relationships_data
        for relationship in relationships_data:
            source_id, target_id = relationship
            session.run("""
                    MATCH (source:Requirements {id: $source_id}), (target:Requirements {id: $target_id})
                    MERGE (source)-[:RELATED_TO]->(target)
                """, {
                "source_id": source_id,
                "target_id": target_id
            })

        # Check if index exists before creation
        if not index_exists(session):
            # Create index for the Requirement label on the text property
            session.run("CREATE INDEX FOR (r:Requirements) ON (r.description)")


def extract_relationships(nodes_data, relationships_data):
    relationships = []

    for node_data in nodes_data:
        node_id = node_data["id"]
        related_node_ids = relationships_data.get(node_id, [])

        for related_node_id in related_node_ids:
            relationships.append((node_id, related_node_id))

    return relationships


def create_nodes_and_relationships(results):
    nodes_to_save = []
    relationships_to_save = []

    # Process nodes
    for node_data in results["nodes"]:
        node_id = node_data["id"]
        description = node_data["properties"]["text"]
        predicted_class = node_data["properties"]["predicted_class"]

        # Add node to save
        nodes_to_save.append({
            "id": node_id,
            "description": description,
            "class": predicted_class
        })

        # Process relationships
        for related_node_id in node_data.get("relationships", []):
            relationships_to_save.append((node_id, related_node_id))

    # Save nodes and relationships to Neo4j
    save_results_to_neo4j_relationship(nodes_to_save, relationships_to_save)

