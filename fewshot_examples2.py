def get_fewshot_examples2(openai_api_key):
    return f""" 
#Identify if this is a functional or non functional requirement  "the balance displayed should be 
accurate and up-to-date, reflecting the most recent transactions”? 
MATCH (r:Requirements)
WHERE r.description contains 'The balance displayed should be accurate and up-to-date, 
reflecting the most recent transactions.'
return r.class
#What are the requirements linked to "the balance displayed should be accurate 
and up-to-date, reflecting the most recent transactions”?
MATCH (main:Requirements)-[r]->(linked:Requirements)
WHERE main.description CONTAINS "The balance displayed should be accurate and up-to-date, 
reflecting the most recent transactions."
RETURN linked
#What is the impact of changing this requirements "the balance displayed should be accurate and up-to-date, 
reflecting the most recent transactions" on other requirements?
MATCH path=(main:Requirements)-[:RELATED_TO*]->(impacted:Requirements)
WHERE main.description CONTAINS "The balance displayed should be accurate and up-to-date, 
reflecting the most recent transactions."
RETURN [node in nodes(path) | node.description] AS ImpactChain, 
[rel in relationships(path) | type(rel)] AS RelationshipClass limit 5
#List all the Functional requirements?
MATCH (r:Requirements)
WHERE r.class = 'Functional'
RETURN r.description AS Description
#List all the Non Functional requirements?
MATCH (r:Requirements)
WHERE r.class = ‘Non-Functional'
RETURN r.description AS Description
#How many Non functional requirements  exist?
MATCH (n:Requirement)
WHERE n.class = 'Non-functional'
RETURN count(n) AS NonFunctionalRequirementsCount
#Explain the impact of the changing requirements?
CALL apoc.ml.openai.embedding(["Explain the impact of the changing requirements?"], 
   "{openai_api_key}") YIELD embedding
MATCH (c:Chunk)
WITH c, gds.similarity.cosine(c.embedding, embedding) AS score
ORDER BY score DESC 
RETURN c.text, score
#How do modifications to the requirement for displaying up-to-date and accurate balances affect  other requirements?
CALL apoc.ml.openai.embedding(["How do modifications to the requirement for displaying up-to-date 
and accurate balances affect  other requirements?"], "{openai_api_key}") YIELD embedding
MATCH (o:Organization {{name:"Microsoft"}})<-[:MENTIONS]-()-[:HAS_CHUNK]->(c)
WITH distinct c, embedding
WITH c, gds.similarity.cosine(c.embedding, embedding) AS score
ORDER BY score DESC 
RETURN c.text, score
#What implications does updating balance display accuracy have on the development timeline?
CALL apoc.ml.openai.embedding(["What implications does updating balance 
display accuracy have on the development timeline?"], "{openai_api_key}") YIELD embedding
MATCH (o:Organization {{name:"Microsoft"}})<-[:MENTIONS]-()-[:HAS_CHUNK]->(c)
WITH distinct c, embedding
WITH c, gds.similarity.cosine(c.embedding, embedding) AS score
ORDER BY score DESC 
RETURN c.text, score
#Can this requirement change lead to a competitive advantage in the market, 
and how might it influence customer acquisition and retention?
CALL apoc.ml.openai.embedding(["Can this requirement change lead to a competitive advantage in the market, 
and how might it influence customer acquisition and retention?"], "{openai_api_key}") YIELD embedding
MATCH (o:Organization {{name:"Microsoft"}})<-[:MENTIONS]-()-[:HAS_CHUNK]->(c)
WITH distinct c, embedding
WITH c, gds.similarity.cosine(c.embedding, embedding) AS score
ORDER BY score DESC 
RETURN c.text, score

When searching for specific information in the text chunks, never use the CONTAINS clause, 
but always use the apoc.ml.openai.embedding
and gds.similarity.cosine functions as shown in the examples.
When returning text chunks, always return exactly three chunks, no more, no less.
Remember, instead of using CONTAINS to find information within text chunks use the apoc.ml.openai.embedding 
and gds.similarity.cosine functions.
"""
