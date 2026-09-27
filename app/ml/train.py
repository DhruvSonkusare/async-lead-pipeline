import os
import sys

# Allow running as `python app/ml/train.py` directly (not just
# `python -m app.ml.train`) by putting the project root on the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.config import get_settings
import pandas as pd
import pickle

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def train_model():
    """Train and evaluate intent classification model"""

    settings = get_settings()

    # Load data
    df = pd.read_csv(settings.get_training_data_path())

    print(f"Loaded {len(df)} records")
    print(f"Classes: {df['intent'].unique()}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"],
        df["intent"],
        test_size=0.2,
        random_state=42,
        stratify=df["intent"]
    )

    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")

    # Build model
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(max_features=1000)
        ),
        (
            "clf",
            MultinomialNB(alpha=0.1)
        )
    ])

    # Train
    print("\nTraining model...")
    model.fit(X_train, y_train)

    # Predict test data
    predictions = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, predictions)

    print("\n========== MODEL RESULTS ==========")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Precision / Recall / F1
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # Save model
    model_path = settings.get_model_path()

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"\nModel saved to: {model_path}")

    return model


if __name__ == "__main__":
    train_model()