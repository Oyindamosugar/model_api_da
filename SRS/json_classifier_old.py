import tensorflow as tf
from transformers import BertTokenizer
import re

# Step 1: Load the Saved Model
model_path = 'model'  # Update this to the path of your saved model
loaded_model = tf.keras.models.load_model(model_path)

# Step 2: Initialize the Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def encode_for_prediction(text, tokenizer):

    encoding = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="tf")
    return encoding

# Function to predict the class of a requirement
def predict_requirement_class(text, tokenizer, model):
    encoding = encode_for_prediction(text, tokenizer)
    predict_input = {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"]
    }
    predictions = model.predict(predict_input)
    probabilities = tf.sigmoid(predictions.logits).numpy()
    predicted_class = 'Functional' if (probabilities > 0.5).astype(int)[0][0] == 1 else 'Non-functional'
    return predicted_class

# Step 3: Read and Extract Requirements from File
file_path = 'software_requirements.txt'  # Path to your file containing requirements.txt
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()

# Assuming each requirement block starts with a digit and dot and ends before a new digit and dot
requirements_blocks = re.split(r'\n(?=\d+\.)', content)

for block in requirements_blocks:
    # Extract the main requirement using the first sentence or identifiable pattern
    main_requirement_match = re.match(r'^\d+\..+?(?=\n)', block, re.DOTALL)
    if main_requirement_match:
        main_requirement = main_requirement_match.group(0)
        print(main_requirement)
        # Predict its class
        #main_class = predict_requirement_class(main_requirement, tokenizer, loaded_model)
        #print(f"Main Requirement: {main_requirement}\nPredicted Class: {main_class}\n")
    else:
        print("No main requirement found in the block.")
