from langchain_ollama import ChatOllama

MODEL_NAME = "gemma3:4b"

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0
)

if __name__ == "__main__":
    print("Testing Ollama...")

    response = llm.invoke(
        "Say hello and tell me that you are working."
    )

    print("\nResponse:")
    print(response.content)