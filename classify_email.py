# classify_email.py
import joblib
import sys
import os
import argparse
from feature_utils import extract_numeric_features

MODEL_PATH = "phishing_model.joblib"

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

if __name__ == "__main__":
    main()
