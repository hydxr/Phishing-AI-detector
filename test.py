import pandas as pd
import numpy as np
import re
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam


from flask import Flask, request, jsonify, render_template

MODEL_PATH = "phishing_model.keras"
VECTORIZER_PATH = "vectorizer.pkl"
DATASET_PATH = "C:/Users/hp/Downloads/ARCHIVE/Phishing_Email.csv" 

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = text.strip()
    return text

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    print("Training model...")

    data = pd.read_csv(DATASET_PATH)
    if 'Email Text' not in data.columns or 'Email Type' not in data.columns:
        raise ValueError("Dataset must have 'Email Text' and 'Email Type' columns.")

    data['Email Text'] = data['Email Text'].fillna("").apply(preprocess_text)
    data = data.dropna(subset=['Email Type'])

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data['Email Type'])

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(data['Email Text']).toarray()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

    model.save(MODEL_PATH)
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
    print("Model trained and saved.")
else:
    print("Using pre-trained model.")


print("Loading model and vectorizer for the server...")
model = load_model(MODEL_PATH)
with open(VECTORIZER_PATH, 'rb') as f:
    vectorizer = pickle.load(f)
print("Model and vectorizer loaded.")


def check_email(email_text):
    """Analyzes email text and returns a dictionary."""
    processed_text = preprocess_text(email_text)
    vectorized_text = vectorizer.transform([processed_text]).toarray()
    prediction = model.predict(vectorized_text)[0][0]

    if prediction > 0.7:
        label = "PHISHING DETECTED"
        css_class = "phishing"
    elif prediction < 0.3:
        label = "NO THREAT DETECTED"
        css_class = "safe"
    else:
        label = "UNCERTAIN THREAT LEVEL"
        css_class = "uncertain"

    return {
        "status": "success",
        "label": label,
        "confidence": float(prediction * 100),
        "css_class": css_class
    }


app = Flask(__name__, static_url_path='', static_folder='.')


@app.route('/')
def home():
   
    return app.send_static_file('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    try:
    
        data = request.get_json()
        email_text = data.get("email_text")

        if not email_text:
            return jsonify({"status": "error", "message": "No email text provided."}), 400

        
        result = check_email(email_text)
        
        
        return jsonify(result)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"status": "error", "message": "Internal server error."}), 500


if __name__ == "__main__":
    print("\nStarting the Phishing Detector web server...")
    print("Open this link in your browser: http://127.0.0.1:5000")
    app.run(debug=False) 