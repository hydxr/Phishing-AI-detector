import pandas as pd
import numpy as np
import re
import os
import pickle
import tkinter as tk
from tkinter import messagebox, scrolledtext
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

# Paths
MODEL_PATH = "phishing_model.keras"
VECTORIZER_PATH = "vectorizer.pkl"
DATASET_PATH = "C:/Users/hp/Downloads/ARCHIVE/Phishing_Email.csv"

# Preprocessing
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = text.strip()
    return text

# Train model if missing
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

# Load
model = load_model(MODEL_PATH)
with open(VECTORIZER_PATH, 'rb') as f:
    vectorizer = pickle.load(f)

# Prediction logic
def predict_email():
    email_text = email_input.get("1.0", tk.END).strip()
    if not email_text:
        messagebox.showwarning("Input Required", "Please paste or type an email!")
        return

    processed_text = preprocess_text(email_text)
    vectorized_text = vectorizer.transform([processed_text]).toarray()
    prediction = model.predict(vectorized_text)[0][0]

    if prediction > 0.7:
        label = "Phishing ⚠️"
        color = "red"
    elif prediction < 0.3:
        label = "Not Phishing ✅"
        color = "lime"
    else:
        label = "Uncertain ⚠️"
        color = "orange"

    label_result.config(
        text=f"{label}\nConfidence: {prediction * 100:.2f}%",
        fg=color
    )

def clear_text():
    email_input.delete("1.0", tk.END)
    label_result.config(text="")

# --- GUI START ---
window = tk.Tk()
window.title("🛡️ Cyber Phishing Detector")
window.geometry("750x600")
window.configure(bg="#0e0e0e")

# Fonts and Colors
title_font = ("Consolas", 20, "bold")
label_font = ("Consolas", 12)
btn_font = ("Consolas", 12, "bold")

# Title
title = tk.Label(window, text="🛡️ CYBER PHISHING DETECTOR 🛡️", font=title_font, bg="#0e0e0e", fg="#00FF88")
title.pack(pady=20)

# Instruction
tk.Label(window, text="Paste Email Below:", font=label_font, bg="#0e0e0e", fg="white").pack()

# Email input
email_input = scrolledtext.ScrolledText(window, width=90, height=15, font=("Courier New", 10), bg="#1e1e1e", fg="#00FF88", insertbackground="white", borderwidth=2, relief="solid")
email_input.pack(pady=10)

# Buttons
button_frame = tk.Frame(window, bg="#0e0e0e")
button_frame.pack()

btn_style = {"font": btn_font, "padx": 20, "pady": 8, "bd": 0}

predict_btn = tk.Button(button_frame, text="🚨 DETECT", command=predict_email, bg="#00FF88", fg="black", activebackground="#1f1f1f", activeforeground="white", **btn_style)
predict_btn.grid(row=0, column=0, padx=10)

clear_btn = tk.Button(button_frame, text="🧹 CLEAR", command=clear_text, bg="#444", fg="white", activebackground="#1f1f1f", activeforeground="white", **btn_style)
clear_btn.grid(row=0, column=1, padx=10)

# Result
label_result = tk.Label(window, text="", font=("Consolas", 14, "bold"), bg="#0e0e0e")
label_result.pack(pady=20)

# Start GUI loop
window.mainloop()
