import streamlit as st
import joblib
import os

MODEL_PATH = "phishing_model.joblib"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None

st.title("Phishing Email Detector (Demo)")

model = load_model()

if model is None:
    st.error("Model file not found. Please run train_phishing.py first to generate phishing_model.joblib.")
else:
    text = st.text_area("Paste email text here")
    if st.button("Classify"):
        cleaned = text.strip()
        if len(cleaned) < 10:
            st.warning("Please enter at least 10 characters of email text.")
        else:
            pred = model.predict([cleaned])[0]
            probs = model.predict_proba([cleaned])[0]
            confidence = max(probs) * 100

            st.write("**Prediction:**", "PHISHING" if pred == 1 else "SAFE")
            st.write(f"**Safe:** {probs[0]*100:.2f}%")
            st.write(f"**Phishing:** {probs[1]*100:.2f}%")

            if confidence < 60:
                st.info("Low confidence prediction — treat this result with caution.")
