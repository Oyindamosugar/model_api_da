from fastapi import FastAPI, UploadFile, File, Form
from transformers import BertTokenizer
import tensorflow as tf
import re
import json
import os

app = FastAPI()

model = tf.keras.Model(inputs={"a": tf.keras.Input(shape=(10,))}, outputs={"b": tf.keras.Input(shape=(10,))})
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

# Load the Saved Model
model_path = 'model'  # Update this to the path of your saved model
# loaded_model = tf.keras.models.load_model(model_path)
loaded_model = tf.keras.optimizers.legacy.Adam(model_path)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


@app.post("/classify/")
async def classify_requirements(file: UploadFile = File(...), output_path: str = Form(...)):
    # Read and Extract Requirements from File
    content = await file.read()
    content = content.decode('utf-8')

    # Use regular expression to remove the "Acceptance Criteria:" prefix from the content
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

    # Save results to the specified JSON file
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, indent=4)

    return {"message": f"Classification results saved to {output_path}"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def readiness_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=int(os.environ.get("PORT", 8000)), host="0.0.0.0")
