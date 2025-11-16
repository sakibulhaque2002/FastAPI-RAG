import json
from numpy.linalg import norm
import numpy as np
import requests
from bert_score import score
from utils.embed import get_embeddings

# -----------------------------
# Load test cases
# -----------------------------
with open("test_cases.json", "r", encoding="utf-8") as f:
    test_cases = json.load(f)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (norm(a) * norm(b)))

THRESHOLD_COSINE = 0.80
THRESHOLD_BERTSCORE = 0.85  # F1 threshold for BERTScore

def run_tests():
    total = len(test_cases)
    passed = 0
    failed = 0

    print("\n========== Chatbot NLP Test Results ==========\n")

    for case in test_cases:
        query = case["query"]
        expected = case["expected"]

        # Call FastAPI chatbot
        try:
            response = requests.post(
                "http://127.0.0.1:8000/query",
                json={"query": query},
                timeout=10
            ).json()
            actual = response.get("answer", "")
        except Exception as e:
            print(f"[ERROR] API unavailable for query: {query}")
            print(e)
            failed += 1
            continue

        # -----------------------------
        # Cosine similarity
        # -----------------------------
        actual_emb, expected_emb = get_embeddings([actual, expected])
        cos_sim = cosine_similarity(actual_emb, expected_emb)

        # -----------------------------
        # BERTScore
        # -----------------------------
        P, R, F1 = score([actual], [expected], lang="en", rescale_with_baseline=True)
        bert_sim = F1[0].item()

        # -----------------------------
        # Decide pass/fail
        # -----------------------------
        # Both metrics must exceed thresholds to PASS
        if cos_sim >= THRESHOLD_COSINE and bert_sim >= THRESHOLD_BERTSCORE:
            status = "PASS"
            passed += 1
        else:
            status = "FAIL"
            failed += 1

        # Print detailed result
        print(f"Query: {query}")
        print(f"Expected: {expected}")
        print(f"Actual: {actual}")
        print(f"Cosine Similarity: {cos_sim:.3f}")
        print(f"BERTScore F1: {bert_sim:.3f} | Result: {status}\n")
        print("-" * 50)

    # Final summary
    print("\n=========== SUMMARY ===========")
    print(f"Total Tests : {total}")
    print(f"Passed      : {passed}")
    print(f"Failed      : {failed}")
    print(f"Accuracy    : {(passed / total) * 100:.2f}%")
    print("================================\n")


if __name__ == "__main__":
    run_tests()
