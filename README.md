# AI Phishing Email Detector

A machine learning-based phishing email classifier trained on 82,000+ real emails.

## Overview
This project uses TF-IDF vectorization + Logistic Regression to classify emails as phishing or safe. Trained on a combined dataset of Enron, Ling, CEAS, Nazario, Nigerian Fraud, and SpamAssassin email corpora.

## Results
- **Dataset size:** 82,486 emails
- **Test accuracy:** 98.2%
- **Precision/Recall/F1:** 0.98 across both classes

## Files
- `train_phishing.py` — trains the model on `phishing_email.csv`
- `classify_email.py` — CLI tool to classify a single email
- `app_streamlit.py` — Streamlit web demo
- `phishing_model.joblib` — trained model (generated after running train_phishing.py, not included in repo)

## Setup
```bash
pip install pandas scikit-learn joblib streamlit
```

Download the dataset from Kaggle: [Phishing Email Dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset)
Place `phishing_email.csv` in the project directory.

## Usage

Train the model:
```bash
python3 train_phishing.py
```

Classify an email via CLI:
```bash
python3 classify_email.py -t "Your email text here"
```

Run the web demo:
```bash
streamlit run app_streamlit.py
```
