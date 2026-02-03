from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def build_model(preprocess_function):
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=preprocess_function)),
        ("clf", LogisticRegression(max_iter=200))
    ])
