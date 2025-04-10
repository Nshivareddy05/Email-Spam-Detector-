import streamlit as st
import joblib
import re
import string
import pandas as pd
import os
import difflib
from datetime import datetime

# ================================
# Load ML model and vectorizer
# ================================
model = joblib.load("spam_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ================================
# CSV File for feedback
# ================================
FEEDBACK_FILE = os.path.join(os.path.dirname(__file__), "user_feedback.csv")

# ================================
# Category Dictionary (Rules-based)
# ================================
categories = {
    # --- ✅ Legitimate Issues ---

    # Delivery/Logistics Issues
    "Order Not Delivered": ["not delivered", "didn’t receive", "haven’t received", "missing order"],
    "Order Delayed": ["delayed", "running late", "late delivery"],
    "Wrong Item Delivered": ["wrong item", "incorrect item", "wrong product"],
    "Damaged Product": ["damaged", "broken", "defective", "torn"],
    "Partial Delivery": ["missing items", "partially delivered"],
    "No Tracking Update": ["no tracking", "can’t track", "tracking not working"],
    "Delivery Agent Issue": ["delivery boy rude", "delivery agent not reachable", "delivery person issue"],
    "Change Delivery Address": ["change address", "update address"],
    "Reschedule Delivery": ["reschedule", "change delivery time"],
    "Pickup Not Done": ["pickup not done", "pickup missed"],

    # Payment & Billing Issues
    "Payment Failed": ["payment failed", "payment not successful", "payment error"],
    "Charged Twice": ["charged twice", "double payment"],
    "Refund Not Received": ["refund not received", "where is my refund"],
    "Overcharged": ["overcharged", "extra amount"],
    "Invoice Request": ["invoice", "bill", "receipt"],
    "Discount Not Applied": ["promo code not working", "discount failed", "coupon not applied"],
    "Amount Not Reflected": ["amount not showing", "money not credited"],

    # Account/Login Issues
    "Login Issues": ["can’t login", "unable to login", "login failed"],
    "Forgot Password": ["forgot password", "reset password"],
    "Account Locked": ["account locked", "too many attempts"],
    "OTP Not Received": ["otp not received", "otp failed", "no otp"],
    "Email/Phone Update Issues": ["change email", "change phone number", "update contact"],

    # App/Technical Issues
    "App Crashing": ["app crashing", "app not opening", "app stopped"],
    "Website Not Loading": ["website not working", "site down"],
    "Blank Screen": ["blank screen", "nothing loads"],
    "Upload Failures": ["can’t upload", "upload failed"],
    "Notification Issues": ["not receiving notifications", "notification not working"],

    # Service Requests
    "Cancel Order": ["cancel order", "order cancellation"],
    "Schedule Pickup": ["schedule pickup", "arrange pickup"],
    "Reschedule Pickup": ["reschedule pickup", "pickup change"],
    "Cancel Pickup": ["cancel pickup"],
    "Pickup Address Change": ["change pickup address"],

    # General Queries
    "How-To/Information": ["how to", "guide", "process", "procedure", "steps"],
    "Talk to Support": ["talk to agent", "human support", "customer care"],
    "Service Availability": ["available in my area", "do you deliver", "service area"],
    "New Registration Help": ["register account", "sign up help"],
    "Feature Request": ["please add", "feature request", "suggestion"],

    # Feedback & Other Issues
    "Complaint": ["complaint", "very bad", "not happy", "disappointed"],
    "Feedback": ["feedback", "review", "rating"],
    "Duplicate Request": ["duplicate request", "sent multiple times"],
    "Uncategorized": [],  # fallback

    # --- 🚨 Fraud & Spam Issues ---

    # Security Threats
    "Phishing Attempt": ["login link", "fake site", "phishing"],
    "Impersonation": ["pretending to be", "fake account"],
    "Malware Link": ["download now", "urgent access"],

    # Scams & Fake Promotions
    "Lottery Scam": ["you won", "congratulations you have won", "lottery"],
    "Fake Discount Offer": ["claim your prize", "special discount for you", "exclusive offer"],
    "Spam Message": ["click this link", "check this out", "visit this site"]
}
# ================================
# Text Cleaning
# ================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ================================
# ML Spam Detection
# ================================
def detect_spam(email_text):
    cleaned = clean_text(email_text)
    email_vector = vectorizer.transform([cleaned])
    prediction = model.predict(email_vector)[0]
    return "🚨 Spam" if prediction == 1 else "✅ Not Spam", prediction

# ================================
# Rule-Based Classification
# ================================
def classify_issue(msg):
    msg_clean = clean_text(msg)
    best_match = None
    max_score = 0

    for category, keywords in categories.items():
        score = 0
        for keyword in keywords:
            keyword_clean = clean_text(keyword)
            if keyword_clean in msg_clean:
                score += 2
            else:
                match_ratio = difflib.SequenceMatcher(None, keyword_clean, msg_clean).ratio()
                if match_ratio > 0.75:
                    score += 1
        if score > max_score:
            max_score = score
            best_match = category

    return best_match if best_match else "No Issues Found ✅"

# ================================
# Save Feedback
# ================================
def save_feedback(email, ml_prediction, rule_category, user_feedback):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "email_text": email,
        "ml_prediction": ml_prediction,
        "rule_category": rule_category,
        "user_feedback": user_feedback
    }
    if not os.path.exists(FEEDBACK_FILE):
        df = pd.DataFrame([entry])
        df.to_csv(FEEDBACK_FILE, index=False)
    else:
        df = pd.read_csv(FEEDBACK_FILE)
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
        df.to_csv(FEEDBACK_FILE, index=False)

# ================================
# Streamlit UI
# ================================
st.set_page_config(page_title="Smart Message Classifier", page_icon="🤖", layout="wide")
st.title("📬 Smart Message Classifier")

email_input = st.text_area("✉️ Paste the email or customer message below:", height=200)

if "prediction_shown" not in st.session_state:
    st.session_state.prediction_shown = False

if st.button("🔍 Analyze Message"):
    if email_input.strip():
        label, prediction = detect_spam(email_input)
        category = classify_issue(email_input)

        st.subheader("🔎 Results")
        st.markdown(f"**Spam Detection (ML):** {label}")
        st.markdown(f"**Detected Category (Rules):** `{category}`")

        st.session_state.ml_prediction = label
        st.session_state.rule_category = category
        st.session_state.prediction_shown = True
    else:
        st.warning("Please enter some message text.")

if st.session_state.prediction_shown:
    st.markdown("#### 🙋 Was this prediction accurate?")
    feedback = st.radio("Your Feedback:", ["Yes", "No"], horizontal=True)
    if st.button("✅ Submit Feedback"):
        save_feedback(email_input, st.session_state.ml_prediction, st.session_state.rule_category, feedback)
        st.success("🎉 Thanks for your feedback. It's been saved.")
        st.session_state.prediction_shown = False

st.markdown("---")
st.markdown("🛡️ Your data stays local and stays safe")
st.sidebar.info("Built with ❤️ by Rahul T")
