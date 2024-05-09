import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the data
df = pd.read_csv('test_data.csv')
requirements = df['requirement'].tolist()
labels = df['type'].values

# Split the dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(requirements, labels, test_size=0.2, random_state=42)

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=1000)  # Adjust max_features as needed
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)



# Initialize and train the Logistic Regression model
logistic_classifier = LogisticRegression(random_state=42)
logistic_classifier.fit(X_train_tfidf, y_train)

# Predict and evaluate the Logistic Regression model
logistic_predictions = logistic_classifier.predict(X_test_tfidf)
logistic_accuracy = accuracy_score(y_test, logistic_predictions)
print(f"Logistic Regression Accuracy: {logistic_accuracy}")
