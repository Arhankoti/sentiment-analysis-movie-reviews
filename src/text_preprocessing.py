import string

# Built-in English stopwords. We do not use NLTK's stopwords here to avoid
# SSL/certificate errors when NLTK tries to download the corpus on some systems
# (e.g. macOS). This list is sufficient for sentiment analysis.
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "shall", "can", "need",
    "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they",
}

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    return " ".join([w for w in words if w not in STOP_WORDS])
