import re
import numpy as np

URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
URGENCY_WORDS = re.compile(r'\b(urgent|immediately|verify|suspend|click here|act now|password|account locked)\b', re.IGNORECASE)

def extract_numeric_features(texts):
    """Extract URL count and urgency-keyword count as extra signals."""
    url_counts = np.array([len(URL_PATTERN.findall(t)) for t in texts]).reshape(-1, 1)
    urgency_counts = np.array([len(URGENCY_WORDS.findall(t)) for t in texts]).reshape(-1, 1)
    return np.hstack([url_counts, urgency_counts])
