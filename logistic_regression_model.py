import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from dataset_loader import get_combined_dataset
from feature_encoders import encode_with_tfidf, encode_with_word2vec


def train_and_save():
    print("Loading combined dataset...")
    combined_train = get_combined_dataset()

    X_text = combined_train["text"]
    y = combined_train["label"]

    # 1. Train / Test Split
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.20, random_state=23
    )

    # 2. Vectorize text with TF-IDF
    print("Fitting TF-IDF Vectorizer...")
    X_train, X_test, vectorizer = encode_with_word2vec(X_train_text, X_test_text)
    # 3. Train Classifier
    print("Training Logistic Regression Model...")
    clf = LogisticRegression(max_iter=10000, random_state=0)
    clf.fit(X_train, y_train)

    # 4. Evaluate Accuracy
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    print(f"\nModel Evaluation -> Accuracy: {acc:.2f}%\n")

    # 5. Save model artifacts to disk
    joblib.dump(vectorizer, "tfidf_vectorizer.joblib")
    joblib.dump(clf, "intent_model.joblib")
    print("Saved 'tfidf_vectorizer.joblib' and 'intent_model.joblib' successfully!")


if __name__ == "__main__":
    train_and_save()