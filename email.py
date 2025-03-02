import streamlit as st
import joblib
import re
import string

# Load the saved model and vectorizer
model = joblib.load("spam_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Function to clean email text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to predict spam or ham
def predict_email(email_text):
    email_text = clean_text(email_text)
    email_vector = vectorizer.transform([email_text])
    prediction = model.predict(email_vector)[0]
    return "🚨 Spam" if prediction == 1 else "✅ Not Spam"

# Streamlit UI
st.title("📩 Email Spam Detector")
st.write("Enter an email below to check if it's spam or not.")

email_input = st.text_area("Paste your email content here:")

if st.button("Check Email"):
    if email_input.strip():
        prediction = predict_email(email_input)
        st.subheader(f"Prediction: {prediction}")
    else:
        st.warning("Please enter some email text!")
