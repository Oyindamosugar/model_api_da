import os
import re

import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from transformers import BertTokenizer

from save_to_neo4j import save_results_to_neo4j

app = FastAPI()

# Load the Saved Model
model_path = 'model'  # Update this to the path of your saved model
loaded_model = tf.keras.models.load_model(model_path)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


@app.post("/classify/")
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
        save_results_to_neo4j(results)
        return {"message": "Classification results saved to Neo4j successfully"}
    except Exception as e:
        return {"error": "Failed to save to Neo4j: {}".format(str(e))}, 500


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def readiness_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=int(os.environ.get("PORT", 8000)), host="0.0.0.0")
