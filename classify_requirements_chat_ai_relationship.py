import os
import re

import json

import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from transformers import BertTokenizer

from fewshot_examples2 import get_fewshot_examples2
from save_to_neo4j_relationship import create_nodes_and_relationships
from driver.neo4j import Neo4jDatabase
from llm.openai import OpenAIChat
from summarize_cypher_result import SummarizeCypherResult
from text2cypher import Text2Cypher
import boto3

# Maximum number of records used in the context
HARD_LIMIT_CONTEXT_RECORDS = 10

neo4j_connection = Neo4jDatabase(
    host=os.environ.get("NEO4J_URL", "neo4j+s://9507ceb2.databases.neo4j.io"),
    user=os.environ.get("NEO4J_USER", "neo4j"),
    password=os.environ.get("NEO4J_PASS", "AUEt0WbwpSdl7LDdqfBc4jfsnwrPIlQFZkY_aKCTD-Y"),
    database=os.environ.get("NEO4J_DATABASE", "neo4j"),
)

# Initialize LLM modules
openai_api_key = os.environ.get("", None)

# Define FastAPI endpoint
app = FastAPI()

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def list_all_files(bucket_name):
    """
    List all files in an S3 bucket.

    :param bucket_name: str - Name of the S3 bucket
    :return: list - A list of file keys in the bucket
    """
    # Create a session using environment variables for AWS credentials
    session = boto3.Session(
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_DEFAULT_REGION', 'eu-north-1')  # Default to us-east-1 if region not set
    )

    s3 = session.client('s3')

    # Initialize list to hold file keys
    file_keys = []

    # Paginate through results to handle large buckets
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name):
        # Check if 'Contents' key is present in the response dictionary
        if 'Contents' in page:
            for obj in page['Contents']:
                file_keys.append(obj['Key'])

    return file_keys

# Usage Example
bucket = 'dissertationartefact'
# Replace with your S3 bucket name
#files = list_all_files(bucket)
#print("Files in S3 bucket:", files)

# Load the Saved Model
model_path = 'dissertationartefact'  # Update this to the path of your saved model
loaded_model = tf.keras.models.load_model(model_path)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


@app.post("/classify/")
async def upload_json_file(file: UploadFile = File(...)):
    # Read and Extract Requirements from File
    content = await file.read()
    content = content.decode('utf-8')

    # Check if content is empty or not in valid JSON format
    if not content.strip():
        return {"error": "Uploaded file is empty"}, 400

    try:
        results = json.loads(content)  # Parse JSON content into Python dictionary
        create_nodes_and_relationships(results)
        return {"message": "Classification results saved to Neo4j successfully"}
    except json.JSONDecodeError as e:
        return {"error": "Failed to parse JSON content: {}".format(str(e))}, 400
    except Exception as e:
        return {"error": "Failed to save to Neo4j: {}".format(str(e))}, 500


@app.post("/classifyV1/")
async def classify_requirements(file: UploadFile = File(...)):
    # Read and Extract Requirements from File
    content = await file.read()
    content = content.decode('utf-8')

    # Use regular expression to remove the "Acceptance Criteria": prefix from the content
    criteria_text = re.sub(r'^Acceptance Criteria:\s*', '', content, flags=re.MULTILINE)

    results = {"nodes": []}  # Initialize a dict with a "nodes" list for storing results

    # Assuming the criteria are separated by new lines
    criteria_list = criteria_text.split('\n')

    for criterion in criteria_list:
        criterion = criterion.strip()
        if criterion:  # Ensure the criterion is not empty
            new_encodings = tokenizer(criterion, padding=True, truncation=True, max_length=128, return_tensors="tf")

            predict_input = {
                "input_ids": new_encodings["input_ids"],
                "attention_mask": new_encodings["attention_mask"]
            }

            if "token_type_ids" in new_encodings:
                predict_input["token_type_ids"] = new_encodings["token_type_ids"]

            predictions = loaded_model.predict(predict_input)
            logits = predictions['logits']
            probabilities = tf.sigmoid(logits).numpy()
            predicted_class = "Functional" if (probabilities > 0.5).astype(int)[0][0] == 0 else "Non-functional"

            results["nodes"].append({
                "id": str(len(results["nodes"]) + 1),
                "labels": predicted_class,
                "properties": {
                    "text": criterion,
                    "predicted_class": predicted_class
                }
            })

    try:
        create_nodes_and_relationships(results)
        return {"message": "Classification results saved to Neo4j successfully"}
    except Exception as e:
        return {"error": "Failed to save to Neo4j: {}".format(str(e))}, 500


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def readiness_check():
    return {"status": "ok"}


@app.websocket("/text2text")
async def websocket_endpoint(websocket: WebSocket):
    async def sendDebugMessage(message):
        await websocket.send_json({"type": "debug", "detail": message})

    async def sendErrorMessage(message):
        await websocket.send_json({"type": "error", "detail": message})

    async def onToken(token):
        delta = token["choices"][0]["delta"]
        if "content" not in delta:
            return
        content = delta["content"]
        if token["choices"][0]["finish_reason"] == "stop":
            await websocket.send_json({"type": "end", "output": content})
        else:
            await websocket.send_json({"type": "stream", "output": content})

        # await websocket.send_json({"token": token})

    await websocket.accept()
    await sendDebugMessage("connected")
    chatHistory = []
    try:
        while True:
            data = await websocket.receive_json()
            if not openai_api_key and not data.get("api_key"):
                raise HTTPException(
                    status_code=422,
                    detail="Please set OPENAI_API_KEY environment variable or send it as api_key in the request body",
                )
            api_key = openai_api_key if openai_api_key else data.get("api_key")

            default_llm = OpenAIChat(
                openai_api_key=api_key,
                model_name=data.get("model_name", "gpt-3.5-turbo-0613"),
            )
            summarize_results = SummarizeCypherResult(
                llm=OpenAIChat(
                    openai_api_key=api_key,
                    model_name="gpt-3.5-turbo-0613",
                    max_tokens=128,
                )
            )

            text2cypher = Text2Cypher(
                database=neo4j_connection,
                llm=default_llm,
                cypher_examples=get_fewshot_examples2(api_key),
            )

            if "type" not in data:
                await websocket.send_json({"error": "missing type"})
                continue
            if data["type"] == "question":
                try:
                    question = data["question"]
                    chatHistory.append({"role": "user", "content": question})
                    await sendDebugMessage("received question: " + question)
                    try:
                        results = text2cypher.run(question, chatHistory)
                        print("results", results)
                    except Exception as e:
                        await sendErrorMessage(str(e))
                        continue
                    if results is None:
                        await sendErrorMessage("Could not generate Cypher statement")
                        continue

                    await websocket.send_json(
                        {
                            "type": "start",
                        }
                    )
                    output = await summarize_results.run_async(
                        question,
                        results["output"][:HARD_LIMIT_CONTEXT_RECORDS],
                        callback=onToken,
                    )
                    chatHistory.append({"role": "system", "content": output})
                    await websocket.send_json(
                        {
                            "type": "answer",
                            "output": output,
                            "generated_cypher": results["generated_cypher"],
                        }
                    )
                except Exception as e:
                    await sendErrorMessage(str(e))
                await sendDebugMessage(output)
    except WebSocketDisconnect:
        print("disconnected")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=int(os.environ.get("PORT", 8000)), host="0.0.0.0")
