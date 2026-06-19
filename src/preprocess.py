"""Text preprocessing module for legal notice classification."""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text: str) -> str:
    """
    Apply full text preprocessing pipeline to a legal notice.

    Args:
        text: Raw input text string.

    Returns:
        Cleaned, preprocessed text string.
    """
    text = re.sub(r'<[^>]+>', '', text)
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)


def preprocess_series(series):
    """
    Apply preprocess_text to a Pandas Series.

    Args:
        series: Pandas Series of raw text strings.

    Returns:
        Pandas Series of preprocessed text strings.
    """
    return series.apply(preprocess_text)
