from sentence_transformers import SentenceTransformer

from pdf_loader import load_all_pdfs
from text_processor import create_context_aware_chunks


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_NAME = "all-MiniLM-L6-v2"


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Embedding model loaded.")


# =========================================================
# GENERATE EMBEDDINGS
# =========================================================

def generate_embeddings(chunks):
    """
    Convert the text of each chunk into a numerical vector.
    """

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        show_progress_bar=True
    )

    return embeddings


# =========================================================
# GENERATE QUERY EMBEDDING
# =========================================================

def generate_query_embedding(query):
    """
    Convert a student's question into the same
    vector space used for document chunks.
    """

    embedding = model.encode(query)

    return embedding


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    pages = load_all_pdfs()

    chunks = create_context_aware_chunks(
        pages
    )

    print("\n==============================")
    print("GENERATING EMBEDDINGS")
    print("==============================")

    print(
        "Total chunks:",
        len(chunks)
    )

    embeddings = generate_embeddings(
        chunks
    )

    print("\n==============================")
    print("EMBEDDING COMPLETED")
    print("==============================")

    print(
        "Number of embeddings:",
        len(embeddings)
    )

    print(
        "Embedding dimensions:",
        len(embeddings[0])
    )

    print("\nFirst embedding:")

    print(
        embeddings[0]
    )