from neo4j import GraphDatabase
import os

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

def insert_record(driver, label, name):
    """Insert a record into the Neo4j database."""
    query = f"CREATE (n:{label} {{name: $name}})"
    with driver.session() as session:
        session.run(query, name=name)

def main():
    try:
        # Connect to Neo4j Aura
        driver = connect_to_neo4j_aura()

        # Example query
        query = "MATCH (n) RETURN COUNT(n) AS node_count"
        records = run_query(driver, query)

        # Print results
        for record in records:
            print(record["node_count"])

        # Insert a dummy record
        insert_record(driver, "Dummy", "Dummy Record")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Close the driver
        driver.close()

if __name__ == "__main__":
    main()