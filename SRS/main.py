import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_csv('test_data.csv')
requirements = df['requirement'].tolist()

# For binary classification, labels should be a 1D array, not one-hot encoded
labels = df['type'].values

# Split the dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(requirements, labels, test_size=0.2, random_state=42)

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def encode_requirements(reqs):
    # Tokenize the text and ensure return_tensors is set to 'tf' to get TensorFlow tensors directly
    return tokenizer(reqs, padding=True, truncation=True, max_length=128, return_tensors="tf")

train_encodings = encode_requirements(X_train)
test_encodings = encode_requirements(X_test)

# Load the pre-trained BERT model for binary classification
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)

# Compile the model with binary cross-entropy loss, which expects labels to be in a 1D array
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

# Convert labels to the correct shape, which should match the logits' shape (None, 1)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Train the model
model.fit(
    {"input_ids": train_encodings["input_ids"], "attention_mask": train_encodings["attention_mask"], "token_type_ids": train_encodings.get("token_type_ids", None)},
    y_train,
    epochs=3,
    batch_size=16,
    validation_split=0.1
)

# Save the model if needed
model.save('model')

# Evaluate the model
print(model.evaluate({"input_ids": test_encodings["input_ids"], "attention_mask": test_encodings["attention_mask"]}, y_test))