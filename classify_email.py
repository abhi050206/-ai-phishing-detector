# classify_email.py
import joblib
import sys
import argparse

MODEL_PATH = "phishing_model.joblib"

def load_model(path=MODEL_PATH):
    return joblib.load(path)

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

    model = load_model()
    label, probs = classify_text(email, model)
    print(f"Prediction: {pretty_label(label)}")
    print(f"Probabilities: {probs}")

if __name__ == "__main__":
    main()
