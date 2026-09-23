import chromadb

from pdf_loader import load_all_pdfs
from text_processor import create_context_aware_chunks
from embeddings import generate_embeddings


# =========================================================
# CONFIGURATION
# =========================================================

CHROMA_PATH = "data/chroma"

COLLECTION_NAME = "university_documents"


# =========================================================
# MAIN
# =========================================================

def main():

    # -----------------------------------------------------
    # 1. Load PDF documents
    # -----------------------------------------------------

    print("Loading PDF documents...")

    pages = load_all_pdfs()

    print(
        f"Pages extracted: {len(pages)}"
    )


    # -----------------------------------------------------
    # 2. Create context-aware chunks
    # -----------------------------------------------------

    print("\nCreating context-aware chunks...")

    chunks = create_context_aware_chunks(
        pages
    )

    print(
        f"Total chunks: {len(chunks)}"
    )


    # -----------------------------------------------------
    # 3. Generate embeddings
    # -----------------------------------------------------

    print("\nGenerating embeddings...")

    embeddings = generate_embeddings(
        chunks
    )

    print(
        f"Generated {len(embeddings)} embeddings."
    )


    # -----------------------------------------------------
    # 4. Connect to ChromaDB
    # -----------------------------------------------------

    print("\nConnecting to ChromaDB...")

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )


    # -----------------------------------------------------
    # 5. Delete old collection
    # -----------------------------------------------------

    print(
        "\nRemoving old collection if it exists..."
    )

    try:

        client.delete_collection(
            name=COLLECTION_NAME
        )

        print("Old collection deleted.")

    except Exception:

        print(
            "No existing collection found."
        )


    # -----------------------------------------------------
    # 6. Create fresh collection
    # -----------------------------------------------------

    print(
        "\nCreating new ChromaDB collection..."
    )

    collection = client.create_collection(
        name=COLLECTION_NAME
    )


    # -----------------------------------------------------
    # 7. Create IDs
    # -----------------------------------------------------

    ids = [
        f"chunk_{i}"
        for i in range(len(chunks))
    ]


    # -----------------------------------------------------
    # 8. Prepare metadata
    # -----------------------------------------------------

    metadatas = []

    for chunk in chunks:

        metadata = {
            "source": str(
                chunk["source"]
            ),

            "page": int(
                chunk["page"]
            ),

            "course_code": (
                chunk["course_code"]
                or ""
            ),

            "course_name": (
                chunk["course_name"]
                or ""
            ),

            "credits": (
                chunk["credits"]
                or ""
            ),

            "section": (
                chunk["section"]
                or ""
            ),

            "unit": (
                chunk["unit"]
                or ""
            )
        }

        metadatas.append(
            metadata
        )


    # -----------------------------------------------------
    # 9. Store everything in ChromaDB
    # -----------------------------------------------------

    print(
        "\nStoring documents in ChromaDB..."
    )

    collection.add(
        ids=ids,
        documents=[
            chunk["text"]
            for chunk in chunks
        ],
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )


    # -----------------------------------------------------
    # 10. Verify
    # -----------------------------------------------------

    print("\n==============================")
    print("VECTOR STORE CREATED")
    print("==============================")

    print(
        "Collection:",
        COLLECTION_NAME
    )

    print(
        "Documents stored:",
        collection.count()
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    main()