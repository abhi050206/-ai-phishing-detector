# train_phishing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Small sample dataset (replace/expand with real dataset like the "phishing dataset" from kaggle)
data = [
    ("Your account has been compromised. Click here to reset your password: http://evil.example", 1),
    ("Please verify your account immediately by clicking the link below", 1),
    ("Meeting notes for today's scrum attached. Let me know your thoughts.", 0),
    ("Invoice for your recent purchase attached. Thank you for shopping with us", 0),
    ("Urgent: Your mailbox is full. Update now to avoid suspension http://phish.example", 1),
    ("Lunch tomorrow? 1 PM works for me.", 0),
    ("You received a secure message from Bank. Please sign in: http://bank.example", 1),
    ("Weekly report attached. Great job everyone!", 0)
]

df = pd.DataFrame(data, columns=["text", "label"])

# Split
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.25, random_state=42)

# Pipeline: TF-IDF + Logistic Regression
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=8000,
        ngram_range=(1,3),
        min_df=2
    )),
    ("clf", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)

# Evaluate
preds = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

# Save model
joblib.dump(pipeline, "phishing_model.joblib")
print("Saved model to phishing_model.joblib")

import numpy as np 
import pandas as pd 
# Input data files are available in the read-only "../input/" directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))





