from pathlib import Path
from pypdf import PdfReader


PDF_DIRECTORY = Path("data/pdfs")


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text and text.strip():
            pages.append({
                "source": pdf_path.name,
                "page": page_number,
                "text": text
            })

    return pages


def load_all_pdfs():
    all_pages = []

    pdf_files = list(PDF_DIRECTORY.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in data/pdfs/")
        return all_pages

    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path.name}")

        pages = extract_text_from_pdf(pdf_path)

        all_pages.extend(pages)

        print(f"Pages extracted: {len(pages)}")

    return all_pages


if __name__ == "__main__":
    pages = load_all_pdfs()

    print("\n==============================")
    print("PDF INGESTION COMPLETED")
    print("==============================")
    print("Total pages extracted:", len(pages))

    for page in pages[:2]:
        print("\n-----------------------------")
        print("Source:", page["source"])
        print("Page:", page["page"])
        print("-----------------------------")
        print(page["text"][:1000])