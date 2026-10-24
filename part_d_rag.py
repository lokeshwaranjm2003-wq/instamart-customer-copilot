
from pathlib import Path
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

POLICY_DIR = Path("data/policies")

# 1. Split policies into overlapping chunks
def chunk_document(text, source, chunk_size=40, overlap=10):
    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        end = min(start + chunk_size, len(words))

        chunks.append({
            "source": source,
            "text": " ".join(words[start:end])
        })

        if end >= len(words):
            break

    return chunks


# 2. Load and chunk all policy files
chunks = []

for path in sorted(POLICY_DIR.glob("*.txt")):
    text = path.read_text(encoding="utf-8")

    chunks.extend(
        chunk_document(text, path.name)
    )

print("=" * 55)
print("PART D: POLICY CHUNKING AND RETRIEVAL")
print("=" * 55)

print("\nTotal chunks:", len(chunks))

counts = Counter(chunk["source"] for chunk in chunks)

print("\nChunks per policy:")
for source, count in sorted(counts.items()):
    print(f"{source}: {count}")


# 3. Fit TF-IDF on policy chunks
vectorizer = TfidfVectorizer()

chunk_texts = [chunk["text"] for chunk in chunks]
chunk_vectors = vectorizer.fit_transform(chunk_texts)


# 4. Retrieve most relevant chunks
def retrieve(query, k=3):
    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        chunk_vectors
    )[0]

    ranked_indices = similarities.argsort()[::-1][:k]

    results = []

    for index in ranked_indices:
        results.append({
            "source": chunks[index]["source"],
            "score": float(similarities[index]),
            "text": chunks[index]["text"]
        })

    return results


# 5. Test retrieval
if __name__ == "__main__":
    test_questions = [
        "When is a full refund allowed for late delivery?",
        "What happens when a refund exceeds INR 1000?",
        "How is customer personal information protected?"
    ]

    for question in test_questions:
        print("\n" + "-" * 55)
        print("QUESTION:", question)

        results = retrieve(question, k=3)

        for rank, result in enumerate(results, start=1):
            print(f"\nRank {rank}")
            print("Source:", result["source"])
            print("Similarity:", round(result["score"], 4))
            print("Text:", result["text"])

    print("\nPOLICY RETRIEVAL COMPLETED")
