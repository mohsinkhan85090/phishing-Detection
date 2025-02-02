import pandas as pd
import zipfile
import os

# Load the dataset (extract zip if needed)
with zipfile.ZipFile("Phising_Detection_Dataset.csv.zip", 'r') as zip_ref:
    zip_ref.extractall('data')

# Load the extracted CSV file
df = pd.read_csv('data/Phising_Detection_Dataset.csv')

# Show the first few rows of the dataset
print(f"Extracted files: {zip_ref.namelist()}")
print(df.head())

# Fill missing 'Phising' values with the mode (most frequent value)
df['Phising'].fillna(df['Phising'].mode()[0], inplace=True)

# Alternatively, drop rows with missing 'Phising' values
# df = df.dropna(subset=['Phising'])

# Check if there are any remaining missing values
print("\nChecking for missing values after handling:")
print(df.isnull().sum())

# Separate the features and the target variable
X = df.drop(['Unnamed: 0', 'Phising'], axis=1)  # Dropping 'Unnamed: 0' as it's just an index
y = df['Phising']

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the size of the train and test sets
print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Train the model using Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("\nModel Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model to a file (optional)
import joblib
joblib.dump(model, 'phishing_model.pkl')

# Optional: Load the model again for predictions
# model = joblib.load('phishing_model.pkl')

# Optional: Make predictions on new data
# Example of new data with the correct number of features (9 features)
# This is just an example, make sure the data matches your feature columns
new_data = [[3, 72, 0, 1, 0, 0, 1, 44, 0]]  # Update this with a valid row of 9 feature values

# Make the prediction
prediction = model.predict(new_data)
print(f"Prediction for the new data: {'Phishing' if prediction[0] == 1 else 'Legitimate'}")
import joblib

# Save the trained model
joblib.dump(model, 'phishing_model.pkl')
print("Model saved as 'phishing_model.pkl'")
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('phishing_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form input data
    num_dots = int(request.form['num_dots'])
    url_length = int(request.form['url_length'])
    # Add other feature inputs here

    # Prepare the data for prediction
    new_data = np.array([[num_dots, url_length]])  # Update this with all features

    # Make prediction
    prediction = model.predict(new_data)
    result = 'Phishing' if prediction[0] == 1 else 'Legitimate'

    return render_template('index.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
