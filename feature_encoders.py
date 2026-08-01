import re
import numpy as np
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


# ------------------------------------------------------------------
# 1. Custom Word2Vec Transformer
# ------------------------------------------------------------------
def _tokenize(text: str):
    """Simple lowercase word tokenizer."""
    return re.findall(r"\w+", text.lower())


class MeanWord2VecVectorizer(BaseEstimator, TransformerMixin):
    """
    Trains a Word2Vec model on input text and transforms sentences
    into fixed-length vectors by averaging token embeddings.
    """

    def __init__(self, vector_size=500, window=5, min_count=1):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.w2v_model = None

    def fit(self, X, y=None):
        tokenized_corpus = [_tokenize(doc) for doc in X]
        self.w2v_model = Word2Vec(
            sentences=tokenized_corpus,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=4,
        )
        return self

    def transform(self, X):
        features = []
        for doc in X:
            tokens = _tokenize(doc)
            vectors = [
                self.w2v_model.wv[word]
                for word in tokens
                if word in self.w2v_model.wv
            ]
            if vectors:
                mean_vec = np.mean(vectors, axis=0)
            else:
                mean_vec = np.zeros(self.vector_size)
            features.append(mean_vec)
        return np.array(features)


# ------------------------------------------------------------------
# 2. Factory Functions to Build Encoders
# ------------------------------------------------------------------
def get_tfidf_encoder(max_features=None, ngram_range=(1, 1)):
    """Returns a configured TF-IDF Vectorizer."""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
    )


def get_word2vec_encoder(vector_size=100, window=5, min_count=1):
    """Returns a configured Mean Word2Vec Vectorizer."""
    return MeanWord2VecVectorizer(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
    )


# ------------------------------------------------------------------
# 3. Direct Helper Functions (Fit & Transform in one step)
# ------------------------------------------------------------------
def encode_with_tfidf(
    X_train, X_test, max_features=None, ngram_range=(1, 1)
):
    """
    Fits TF-IDF on X_train and transforms both X_train and X_test.
    Returns: (X_train_vec, X_test_vec, vectorizer)
    """
    vectorizer = get_tfidf_encoder(
        max_features=max_features, ngram_range=ngram_range
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec, vectorizer


def encode_with_word2vec(
    X_train, X_test, vector_size=100, window=5, min_count=1
):
    """
    Fits Word2Vec on X_train and transforms both X_train and X_test.
    Returns: (X_train_vec, X_test_vec, vectorizer)
    """
    vectorizer = get_word2vec_encoder(
        vector_size=vector_size, window=window, min_count=min_count
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec, vectorizer