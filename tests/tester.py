# tests/tester.py
import json
import requests
from adapters.cosine_adapter import CosineAdapter
from adapters.bertscore_adapter import BERTScoreAdapter
from adapters.gemini_adapter import GeminiAdapter

with open("test_cases.json", "r", encoding="utf-8") as f:
    test_cases = json.load(f)

# -----------------------------
# Choose your similarity metric
# -----------------------------
METRIC = "gemini"

if METRIC.lower() == "gemini":
    similarity_adapter = GeminiAdapter()
elif METRIC.lower() == "cosine":
    similarity_adapter = CosineAdapter()
elif METRIC.lower() == "bertscore":
    similarity_adapter = BERTScoreAdapter()
else:
    raise ValueError("Unsupported metric")

# -----------------------------
# Run tests
# -----------------------------
def run_tests():
    total = len(test_cases)
    passed = 0
    failed = 0

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

        sim = similarity_adapter.compute(actual, expected)
        status = "PASS" if similarity_adapter.check_pass(actual, expected) else "FAIL"

        print(f"Query: {query}")
        print(f"Expected: {expected}")
        print(f"Actual: {actual}")
        print(f"Similarity: {sim:.3f} | Result: {status}\n")
        print("-" * 50)

        if status == "PASS":
            passed += 1
        else:
            failed += 1

    # Summary
    print("\n=========== SUMMARY ===========")
    print(f"Total Tests : {total}")
    print(f"Passed      : {passed}")
    print(f"Failed      : {failed}")
    print(f"Accuracy    : {(passed / total) * 100:.2f}%")
    print("================================\n")


if __name__ == "__main__":
    run_tests()
