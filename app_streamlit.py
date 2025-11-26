import streamlit as st
import joblib

@st.cache_resource
def load_model():
    return joblib.load("phishing_model.joblib")

st.title("Phishing Email Detector (Demo)")
model = load_model()

text = st.text_area("Paste email text here")
if st.button("Classify"):
    if not text.strip():
        st.warning("Please enter text.")
    else:
        pred = model.predict([text])[0]
        probs = model.predict_proba([text])[0]
        st.write("**Prediction:**", "PHISHING" if pred==1 else "SAFE")
        st.write("**Probabilities:**", probs)
