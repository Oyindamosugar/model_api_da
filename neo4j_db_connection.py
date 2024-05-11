from neo4j import GraphDatabase


def connect_to_neo4j_aura():
    uri = "neo4j+s://9507ceb2.databases.neo4j.io:7687"
    user = "neo4j"
    password = "AUEt0WbwpSdl7LDdqfBc4jfsnwrPIlQFZkY_aKCTD-Y"
    driver = GraphDatabase.driver(uri, auth=(user, password))
    return driver


def run_query(driver, query):
    """Execute a Cypher query and return a list of results."""
    with driver.session() as session:
        result = session.run(query)
        return [record for record in result]


def get_db_session():
    return connect_to_neo4j_aura().session()
