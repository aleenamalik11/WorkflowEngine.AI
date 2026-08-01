import json
import joblib


class IntentPredictor:

    def __init__(
        self,
        model_path="intent_model.joblib",
        vectorizer_path="tfidf_vectorizer.joblib",
        mapping_path="intent_mapping.json",
    ):
        print("Loading model artifacts...")
        self.clf = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

        with open(mapping_path, "r") as f:
            raw_mapping = json.load(f)
            # Cast keys from JSON string -> int
            self.id2intent = {int(k): v for k, v in raw_mapping.items()}

    def predict(self, prompt: str):
        prompt_vec = self.vectorizer.transform([prompt])
        predicted_id = int(self.clf.predict(prompt_vec)[0])
        probs = self.clf.predict_proba(prompt_vec)[0]
        confidence = float(probs.max() * 100)

        intent_name = self.id2intent.get(
            predicted_id, f"Unknown ID ({predicted_id})"
        )

        return {
            "prompt": prompt,
            "predicted_id": predicted_id,
            "intent": intent_name,
            "confidence": f"{confidence:.2f}%",
        }


# --- THIS BLOCK RUNS WHEN YOU EXECUTE THE FILE ---
if __name__ == "__main__":
    predictor = IntentPredictor()

    test_prompts = [
        "I have an incorrect charge on my card and would like a refund.",
        "How do I change my PIN number?",
        "Transfer $50 to my savings account.",
        "Show restaurants near me",
    ]

    print("\n" + "=" * 60)
    print("INFERENCE TEST RESULTS")
    print("=" * 60)

    for prompt in test_prompts:
        res = predictor.predict(prompt)
        print(f"Prompt    : {res['prompt']}")
        print(f"Intent    : {res['intent']} (ID: {res['predicted_id']})")
        print(f"Confidence: {res['confidence']}")
        print("-" * 60)