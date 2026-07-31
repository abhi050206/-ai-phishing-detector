# classify_email.py
import joblib
import sys
import os
import re
import argparse
from feature_utils import extract_numeric_features

MODEL_PATH = "phishing_model.joblib"

URGENCY_WORDS = ["urgent", "immediately", "verify your account", "suspended", "act now",
                  "limited time", "click here", "confirm your identity", "unusual activity",
                  "your account will be", "final notice"]

URL_PATTERN = re.compile(r'https?://[^\s]+')
SHORTENER_DOMAINS = ["bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly"]

def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        print(f"Error: Model file '{path}' not found.")
        print("Run 'python3 train_phishing.py' first to generate the model.")
        sys.exit(1)
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"Error: Failed to load model — {e}")
        sys.exit(1)

def classify_text(text, model):
    pred = model.predict([text])[0]
    probs = model.predict_proba([text])[0]
    return pred, probs

def pretty_label(label):
    return "PHISHING" if label == 1 else "SAFE"

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

def main():
    parser = argparse.ArgumentParser(description="Classify an email as phishing or safe.")
    parser.add_argument("-t", "--text", help="Email text to classify", required=False)
    args = parser.parse_args()

    if args.text:
        email = args.text
    else:
        print("Paste the email text and then press Ctrl+D (EOF):")
        email = sys.stdin.read()

    email = email.strip()
    if len(email) < 10:
        print("Error: Email text too short to classify meaningfully (minimum 10 characters).")
        sys.exit(1)

    model = load_model()
    label, probs = classify_text(email, model)
    confidence = max(probs) * 100

    print(f"Prediction: {pretty_label(label)}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Probabilities: SAFE={probs[0]*100:.2f}% | PHISHING={probs[1]*100:.2f}%")

    if confidence < 60:
        print("Note: Low confidence prediction — treat this result with caution.")

    reasons = analyze_reasons(email)
    if reasons:
        print("\nReasons flagged:")
        for r in reasons:
            print(f"  - {r}")
    elif label == 1:
        print("\nReasons flagged: Model detected phishing patterns not captured by heuristic checks (based on learned word patterns from training data).")

if __name__ == "__main__":
    main()
