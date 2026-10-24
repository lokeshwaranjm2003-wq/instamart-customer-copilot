
import mock_llm
from part_d_rag import retrieve

SYSTEM_PROMPT = """
You are a POLICY ASSISTANT for an Instamart support team.
Answer only using the supplied policy context.
Do not invent refund rules, amounts, or exceptions.
If the context does not support an answer, say:
I cannot answer from the available policy documents.
"""

FALLBACK = "I cannot answer from the available policy documents."


def ask_policy(question):
    # Retrieve top 3 relevant policy chunks
    retrieved = retrieve(question, k=3)

    # Build context with source references
    context = "\n\n".join(
        f"Source: {item['source']}\n{item['text']}"
        for item in retrieved
    )

    # Guardrail: do not answer if no relevant chunk exists
    if not retrieved or retrieved[0]["score"] <= 0:
        return {
            "question": question,
            "answer": FALLBACK,
            "sources": [],
            "tokens_used": 0
        }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Policy context:\n{context}\n\n"
                f"Question: {question}"
            )
        }
    ]

    response = mock_llm.chat(messages)
    answer = response.choices[0].message.content

    return {
        "question": question,
        "answer": answer,
        "sources": list(dict.fromkeys(
            item["source"] for item in retrieved
        )),
        "tokens_used": response.usage.total_tokens
    }


if __name__ == "__main__":
    questions = [
        "When is a full refund allowed for late delivery?",
        "What happens if a refund exceeds INR 1000?",
        "How is customer personal information protected?",
        "What is the CEO's favourite movie?"
    ]

    print("\nPART D: POLICY ASSISTANT TEST")

    for question in questions:
        result = ask_policy(question)

        print("\n" + "-" * 55)
        print("Question:", result["question"])
        print("Answer:", result["answer"])
        print("Sources:", result["sources"])
        print("Tokens:", result["tokens_used"])

    print("\nPOLICY ASSISTANT TEST COMPLETED")
