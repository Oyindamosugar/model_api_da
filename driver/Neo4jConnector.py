from neo4j import GraphDatabase, exceptions


class Neo4jConnector:
    def __init__(self, uri, user, password):
        self._uri = uri
        self._user = user
        self._password = password
        self._driver = self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password), encrypted=False)
        # Verify connection
        try:
            self._driver.verify_connectivity()
        except exceptions.ServiceUnavailable:
            raise ValueError(
                "Could not connect to Neo4j database. "
                "Please ensure that the url is correct"
            )
        except exceptions.AuthError:
            raise ValueError(
                "Could not connect to Neo4j database. "
                "Please ensure that the username and password are correct"
            )

    def connect(self):
        """Connect to the Neo4j database."""
        self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))

    def close(self):
        """Close the connection to the Neo4j database."""
        if self._driver is not None:
            self._driver.close()

    def run_query(self, query):
        """Execute a Cypher query and return a list of results."""
        with self._driver.session() as session:
            result = session.run(query)
            return [record for record in result]

    def insert_record(self, label, name):
        """Insert a record into the Neo4j database."""
        query = f"CREATE (n:{label} {{name: $name}})"
        with self._driver.session() as session:
            session.run(query, name=name)

