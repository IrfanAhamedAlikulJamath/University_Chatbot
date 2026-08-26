import chromadb
from sentence_transformers import SentenceTransformer

from pdf_loader import load_all_pdfs
from text_processor import create_context_aware_chunks


MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "university_documents"


def create_vector_store():

    print("Loading PDF documents...")

    pages = load_all_pdfs()

    print("Creating context-aware chunks...")

    chunks = create_context_aware_chunks(pages)

    print("Total chunks:", len(chunks))

    print("\nLoading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    texts = [chunk["text"] for chunk in chunks]

    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        show_progress_bar=True
    )

    print("\nConnecting to ChromaDB...")

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    print("Storing documents in ChromaDB...")

    ids = [
        f"chunk_{i}"
        for i in range(len(chunks))
    ]

    documents = [
        chunk["text"]
        for chunk in chunks
    ]

    metadatas = [
        {
            "source": chunk["source"],
            "page": chunk["page"],
            "course_code": chunk["course_code"] or "",
            "course_name": chunk["course_name"] or "",
            "unit": chunk["unit"] or ""
        }
        for chunk in chunks
    ]

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    print("\n==============================")
    print("VECTOR STORE CREATED")
    print("==============================")

    print("Collection:", COLLECTION_NAME)
    print("Documents stored:", collection.count())


if __name__ == "__main__":
    create_vector_store()