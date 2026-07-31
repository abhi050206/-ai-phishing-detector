import streamlit as st
import joblib
import re
from feature_utils import extract_numeric_features
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

URGENCY_WORDS = ["urgent", "immediately", "verify your account", "suspended", "act now",
                  "limited time", "click here", "confirm your identity", "unusual activity",
                  "your account will be", "final notice"]

URL_PATTERN = re.compile(r'https?://[^\s]+')
SHORTENER_DOMAINS = ["bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly"]

def analyze_reasons(text):
    reasons = []
    lower_text = text.lower()

    urls = URL_PATTERN.findall(text)
    if len(urls) >= 1:
        reasons.append(f"Contains {len(urls)} URL(s)")

    shortened = [u for u in urls if any(domain in u for domain in SHORTENER_DOMAINS)]
    if shortened:
        reasons.append(f"Contains {len(shortened)} shortened URL(s) — common phishing evasion tactic")

    matched_urgency = [w for w in URGENCY_WORDS if w in lower_text]
    if matched_urgency:
        reasons.append(f"Urgency/pressure language detected: {', '.join(matched_urgency[:3])}")

    if "verify" in lower_text and "account" in lower_text:
        reasons.append("Contains 'verify account' pattern — common credential-harvesting phrase")

    if re.search(r'\$\d+|\bwon\b|\bprize\b|\breward\b', lower_text):
        reasons.append("Contains money/prize-related language")

    exclaim_count = text.count("!")
    if exclaim_count >= 2:
        reasons.append(f"Excessive punctuation ({exclaim_count} exclamation marks) — common in spam/phishing")

    return reasons

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

            reasons = analyze_reasons(cleaned)
            if reasons:
                st.write("**Reasons flagged:**")
                for r in reasons:
                    st.write(f"- {r}")
            elif pred == 1:
                st.write("**Reasons flagged:** Model detected phishing patterns not captured by heuristic checks (based on learned word patterns from training data).")
