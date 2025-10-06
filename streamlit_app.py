# ===============================
# streamlit_app.py
# AI Chatbot Tutor - Multi-turn Interactive Chat
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Optional XAI
try:
    import shap
    HAS_SHAP = True
except:
    HAS_SHAP = False

st.set_page_config(page_title="AI Chatbot Tutor", page_icon="🤖", layout="wide")
st.title("🧠 AI Chatbot Tutor - Multi-turn Chat")

st.markdown("""
Welcome to your personalized AI tutor! Type a sentence, and the tutor will provide feedback.
Your conversation history is maintained so you can practice interactively.
""")

# -------------------------------
# Load trained model + preprocessor
# -------------------------------
try:
    rf_model = joblib.load("artifacts/RandomForest_model.joblib")
    preprocessor = joblib.load("artifacts/preprocessor.joblib")
    st.success("Model and preprocessor loaded ✅")
except:
    rf_model = None
    preprocessor = None
    st.warning("No model found. Using random feedback for demonstration.")

# -------------------------------
# Session state for multi-turn chat
# -------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------------
# Feature extraction from text
# -------------------------------
def extract_features_from_text(text):
    """
    Converts text into numeric features for the model.
    Replace with your actual feature engineering used in training.
    """
    text = text.lower()
    features = {}
    features['length'] = len(text)
    features['num_words'] = len(text.split())
    features['num_commas'] = text.count(',')
    features['num_excl'] = text.count('!')
    return pd.DataFrame([features])

# -------------------------------
# Get feedback from text input
# -------------------------------
def get_feedback_from_text(text):
    X_input = extract_features_from_text(text)
    shap_text = ""
    if rf_model and preprocessor:
        X_proc = preprocessor.transform(X_input)
        prob = rf_model.predict_proba(X_proc)[:,1][0]
        if prob < 0.4:
            feedback = f"It seems you're struggling (predicted success {prob:.2f}). Hint: Focus on grammar."
        elif prob < 0.8:
            feedback = f"You're doing okay (predicted success {prob:.2f}). Try this practice sentence."
        else:
            feedback = f"Great! Likely success (predicted {prob:.2f}). Let's try a harder sentence!"
        # SHAP explanation
        if HAS_SHAP:
            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer.shap_values(X_proc)
            shap_text = " | ".join([f"{feat}:{val:.2f}" for feat,val in zip(X_input.columns, shap_values[1][0])])
        return feedback, shap_text
    else:
        feedback_options = [
            "Excellent work! 🎉",
            "Check your grammar ✏️",
            "Try again 🔄",
            "You're improving! 🚀"
        ]
        return np.random.choice(feedback_options), shap_text

# -------------------------------
# Streamlit input box
# -------------------------------
user_input = st.text_input("Type your sentence:")

if st.button("Send"):
    if user_input.strip() != "":
        feedback, shap_info = get_feedback_from_text(user_input)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Tutor", feedback))
        if shap_info:
            st.session_state.chat_history.append(("XAI", shap_info))

# -------------------------------
# Display chat history
# -------------------------------
st.subheader("Conversation")
for speaker, message in st.session_state.chat_history:
    if speaker == "You":
        st.markdown(f"**You:** {message}")
    elif speaker == "Tutor":
        st.markdown(f"**Tutor:** {message}")
    elif speaker == "XAI":
        st.markdown(f"*Feature contributions:* {message}")

# -------------------------------
# Optional reset
# -------------------------------
if st.button("Reset Chat"):
    st.session_state.chat_history = []
    st.experimental_rerun()

st.markdown("""
---
### Notes:
- Replace `extract_features_from_text` with your actual feature extraction for your trained model.
- The AI tutor now supports multi-turn conversations and shows SHAP explanations for each turn (if SHAP installed).
- Deploy to [Streamlit Cloud](https://streamlit.io/cloud) for free online use.
""")
