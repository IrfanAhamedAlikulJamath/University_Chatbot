from retriever import search_documents
from llm import llm


def build_context(results):
    context_parts = []

    for i, result in enumerate(results, start=1):
        metadata = result["metadata"]
        document = result["document"]

        context_parts.append(
            f"""
SOURCE {i}
Source: {metadata.get("source", "")}
Page: {metadata.get("page", "")}
Course: {metadata.get("course_name", "")}
Section: {metadata.get("section", "")}
Unit: {metadata.get("unit", "")}

Content:
{document}
"""
        )

    return "\n".join(context_parts)


def generate_answer(question, results):
    context = build_context(results)

    prompt = f"""
You are a university assistance chatbot.

Your task is to answer the student's question using ONLY the
information explicitly provided in the CONTEXT.

IMPORTANT RULES:

1. Do not use your own knowledge.
2. Do not guess or infer missing information.
3. Do not combine values from different courses.
4. Always identify the course name and course code before
   answering a course-specific question.
5. If the question asks for credits, use the value explicitly
   labeled "Credits".
6. Do not confuse L, T, P, EL, Credits, or Total Marks.
7. If multiple courses appear in the context, use only the
   course that matches the student's question.
8. If the requested information is not clearly present,
   say:
   "I couldn't find this information in the provided
   university documents."
9. Give a direct, natural-language answer.
10. For simple factual questions, answer in one sentence.

CONTEXT:
{context}

STUDENT QUESTION:
{question}

Before answering, verify that the answer comes from the
correct course in the context.

ANSWER:
"""

    response = llm.invoke(prompt)

    return response.content

def main():
    print("==============================")
    print("UNIVERSITY ASSISTANCE CHATBOT")
    print("==============================")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Student: ").strip()

        if question.lower() == "exit":
            print("\nChatbot: Goodbye!")
            break

        if not question:
            continue

        print("\nSearching university documents...")

        results = search_documents(question)

        print(f"Retrieved {len(results)} relevant chunks.")

        print("\nGenerating answer...")

        answer = generate_answer(question, results)

        print("\nChatbot:", answer)
        print()


if __name__ == "__main__":
    main()