# AI Phishing Email Detector

A machine learning-based phishing email detector using TF-IDF vectorization and Logistic Regression.

## Features
- Train a custom phishing detection model
- Command-line interface for quick email classification
- Optional Streamlit web UI
- High accuracy detection using NLP techniques

## Installation

1. Clone the repository:
```bash
git clone https://github.com/abhi050206/ai-phishing-detector.git
cd ai-phishing-detector
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Train the Model
```bash
python train_phishing.py
```

### Classify Emails (CLI)
```bash
python classify_email.py -t "Your suspicious email text here"
```

### Run Web Interface (Optional)
```bash
streamlit run app_streamlit.py
```

## Project Structure
```
.
├── train_phishing.py      # Model training script
├── classify_email.py      # CLI classification tool
├── app_streamlit.py       # Web UI (optional)
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

## Model Details
- **Algorithm**: Logistic Regression
- **Vectorization**: TF-IDF (1-3 grams)
- **Features**: 8000 max features

## License
MIT License
