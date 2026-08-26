import chromadb
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "university_documents"


def search_documents(query, number_of_results=5):

    model = SentenceTransformer(MODEL_NAME)

    query_embedding = model.encode(query)

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=number_of_results
    )

    return results


if __name__ == "__main__":

    query = input("\nEnter your question: ")

    results = search_documents(query)

    print("\n==============================")
    print("RETRIEVAL RESULTS")
    print("==============================")

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances),
        start=1
    ):

        print("\n-----------------------------")
        print(f"Result {i}")
        print("-----------------------------")

        print("Source:", metadata["source"])
        print("Page:", metadata["page"])
        print("Course Code:", metadata["course_code"])
        print("Course Name:", metadata["course_name"])
        print("Unit:", metadata["unit"])
        print("Distance:", distance)

        print("\nText:")
        print(document)