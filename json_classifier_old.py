import os

import tensorflow as tf
from transformers import BertTokenizer
import re
import json
# Step 1: Load the Saved Model
import boto3
def list_top_level_prefixes(bucket_name):
    """
    List top-level 'folder' names (prefixes) in an S3 bucket, retaining the trailing slashes.

    :param bucket_name: str - Name of the S3 bucket
    :return: list - A list of top-level prefix names with trailing slashes
    """
    # Create a session using environment variables for AWS credentials
    session = boto3.Session(
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_DEFAULT_REGION', 'eu-north-1')  # Default to us-east-1 if region not set
    )

    s3 = session.client('s3')

    # Initialize a list to hold unique top-level prefixes
    prefixes = []

    # Use a paginator to handle potential large lists of objects
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name, Delimiter='/'):
        # 'CommonPrefixes' contains the folder/prefix information at the top level
        for prefix in page.get('CommonPrefixes', []):
            # Add the prefix as is, with the trailing slash
            prefixes.append(prefix['Prefix'])

    return prefixes


# Usage Example
bucket = 'dissertationartefact'  # Replace with your S3 bucket name
top_level_prefixes = list_top_level_prefixes(bucket)
for prefix in top_level_prefixes:
    print(prefix)  # This will print each prefix with the trailing slash

model_path = prefix # Update this to the path of your saved model
loaded_model = tf.keras.models.load_model(model_path)

# Step 3: Read and Extract Requirements from File
file_path = 'software_requirements.txt'  # Path to your file containing requirements.txt
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()

# Use regular expression to remove the "Acceptance Criteria:" prefix from the content
criteria_text = re.sub(r'^Acceptance Criteria:\s*', '', content, flags=re.MULTILINE)

results = {"nodes": []}  # Initialize a dict with a "nodes" list for storing results

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

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

# Save results to a JSON file
output_file_path = 'classification_results.json'
with open(output_file_path, 'w', encoding='utf-8') as json_file:
    json.dump(results, json_file, indent=4)

print(f"Results saved to {output_file_path}")