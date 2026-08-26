from sentence_transformers import SentenceTransformer

from pdf_loader import load_all_pdfs
from text_processor import create_context_aware_chunks


MODEL_NAME = "all-MiniLM-L6-v2"


def generate_embeddings(chunks):
    model = SentenceTransformer(MODEL_NAME)

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        show_progress_bar=True
    )

    return embeddings


if __name__ == "__main__":
    pages = load_all_pdfs()

    chunks = create_context_aware_chunks(pages)

    print("\n==============================")
    print("GENERATING EMBEDDINGS")
    print("==============================")

    print("Total chunks:", len(chunks))

    embeddings = generate_embeddings(chunks)

    print("\n==============================")
    print("EMBEDDING COMPLETED")
    print("==============================")

    print("Number of embeddings:", len(embeddings))
    print("Embedding dimensions:", len(embeddings[0]))

    print("\nFirst embedding:")
    print(embeddings[0])