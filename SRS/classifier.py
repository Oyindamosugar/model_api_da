import tensorflow as tf
from transformers import BertTokenizer

# Step 1: Load the Saved Model
model_path = 'model'  # Update this to the path of your saved model
loaded_model = tf.keras.models.load_model(model_path)

# Step 2: Prepare the Unseen Data
new_texts = ["The system should display the current balance of the customer's account when they log in."]  # Example new text data

# Initialize the tokenizer with the configuration used during training
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize and encode the new text data with the same parameters used during training
def encode_for_prediction(new_texts, tokenizer):

    # Tokenize and encode the new text data
    new_encodings = tokenizer(new_texts, padding=True, truncation=True, max_length=128, return_tensors="tf")

    return new_encodings

# Prepare the new text data
new_encodings = encode_for_prediction(new_texts, tokenizer)

# Prepare input as a dictionary to match the expected input format
predict_input = {
    "input_ids": new_encodings["input_ids"],
    "attention_mask": new_encodings["attention_mask"]
}

# Add "token_type_ids" if your model and tokenizer include them
if "token_type_ids" in new_encodings:
    predict_input["token_type_ids"] = new_encodings["token_type_ids"]

# Use the prepared input for prediction
predictions = loaded_model.predict(predict_input)

# Extract logits from the predictions dictionary
logits = predictions['logits']

# If your model outputs logits, convert these to probabilities
probabilities = tf.sigmoid(logits).numpy()

# Apply a threshold to determine class labels
predicted_classes = (probabilities > 0.5).astype(int)

# Print the predicted class for each input text
for text, pred_class in zip(new_texts, predicted_classes):
    print(f"Text: '{text}'\nPredicted class: {'Functional' if pred_class[0] == 0 else 'Non-functional'}\n")
