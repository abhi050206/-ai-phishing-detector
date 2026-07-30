# train_phishing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
from feature_utils import extract_numeric_features

# Load real dataset (82,487 emails from Kaggle: Enron, Ling, CEAS, Nazario, Nigerian, SpamAssassin)
df = pd.read_csv("phishing_email.csv")
df = df.dropna(subset=["text_combined", "label"])
df = df.rename(columns={"text_combined": "text"})
print(f"Total samples: {len(df)}")
print(df["label"].value_counts())

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

text_features = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=8000,
        ngram_range=(1, 3),
        min_df=2
    ))
])

numeric_features = Pipeline([
    ("extract", FunctionTransformer(extract_numeric_features, validate=False))
])

combined_features = FeatureUnion([
    ("text", text_features),
    ("numeric", numeric_features)
])

pipeline = Pipeline([
    ("features", combined_features),
    ("clf", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)

# Evaluate
preds = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))
print("Confusion Matrix:")
print(confusion_matrix(y_test, preds))

# Save model
joblib.dump(pipeline, "phishing_model.joblib")
print("Saved model to phishing_model.joblib")
