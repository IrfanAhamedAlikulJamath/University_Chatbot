import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf_loader import load_all_pdfs


COURSE_PATTERN = re.compile(
    r"\b([A-Z]{2,}[A-Z0-9]*\d{4,})\s+([A-Z][A-Z\s&–-]+?)\s+L\s+T\s+P",
    re.IGNORECASE
)

UNIT_PATTERN = re.compile(
    r"\bUNIT\s+([1-5])\b",
    re.IGNORECASE
)


def detect_course(text):
    match = COURSE_PATTERN.search(text)

    if match:
        course_code = match.group(1).strip()
        course_name = match.group(2).strip()

        return course_code, course_name

    return None, None


def create_enriched_text(course_code, course_name, unit, chunk):

    context_parts = []

    if course_code:
        context_parts.append(f"Course Code: {course_code}")

    if course_name:
        context_parts.append(f"Course Name: {course_name}")

    if unit:
        context_parts.append(f"Unit: {unit}")

    context = "\n".join(context_parts)

    if context:
        return f"{context}\n\n{chunk}"

    return chunk


def create_context_aware_chunks(pages):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=300,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []

    current_course_code = None
    current_course_name = None
    current_unit = None

    for page in pages:

        text = page["text"]

        course_code, course_name = detect_course(text)

        if course_code:
            current_course_code = course_code
            current_course_name = course_name
            current_unit = None

        units = list(UNIT_PATTERN.finditer(text))

        if not units:

            page_chunks = text_splitter.split_text(text)

            for chunk in page_chunks:

                enriched_text = create_enriched_text(
                    current_course_code,
                    current_course_name,
                    current_unit,
                    chunk
                )

                chunks.append({
                    "source": page["source"],
                    "page": page["page"],
                    "course_code": current_course_code,
                    "course_name": current_course_name,
                    "unit": current_unit,
                    "text": enriched_text
                })

        else:

            for i, unit_match in enumerate(units):

                current_unit = f"UNIT {unit_match.group(1)}"

                start = unit_match.start()

                if i + 1 < len(units):
                    end = units[i + 1].start()
                else:
                    end = len(text)

                unit_text = text[start:end]

                page_chunks = text_splitter.split_text(unit_text)

                for chunk in page_chunks:

                    enriched_text = create_enriched_text(
                        current_course_code,
                        current_course_name,
                        current_unit,
                        chunk
                    )

                    chunks.append({
                        "source": page["source"],
                        "page": page["page"],
                        "course_code": current_course_code,
                        "course_name": current_course_name,
                        "unit": current_unit,
                        "text": enriched_text
                    })

    return chunks


if __name__ == "__main__":

    pages = load_all_pdfs()

    chunks = create_context_aware_chunks(pages)

    print("\n==============================")
    print("CONTEXT-AWARE CHUNKING")
    print("==============================")

    print("Total pages:", len(pages))
    print("Total chunks:", len(chunks))

    for i, chunk in enumerate(chunks[:5], start=1):

        print("\n-----------------------------")
        print("Chunk:", i)
        print("Source:", chunk["source"])
        print("Page:", chunk["page"])
        print("Course Code:", chunk["course_code"])
        print("Course Name:", chunk["course_name"])
        print("Unit:", chunk["unit"])
        print("-----------------------------")

        print(chunk["text"])