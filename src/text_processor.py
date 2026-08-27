import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf_loader import load_all_pdfs


# =========================================================
# PATTERNS
# =========================================================

COURSE_PATTERN = re.compile(
    r"\b([A-Z]{2,}[A-Z0-9]*\d{4,})\s+([A-Z][A-Z\s&–-]+?)\s+L\s+T\s+P",
    re.IGNORECASE
)

UNIT_PATTERN = re.compile(
    r"\bUNIT\s+([1-5])\b",
    re.IGNORECASE
)

COURSE_OUTCOMES_PATTERN = re.compile(
    r"\bCOURSE\s+OUTCOMES\b",
    re.IGNORECASE
)

REFERENCES_PATTERN = re.compile(
    r"\bTEXT\s*/\s*REFERENCE\s+BOOKS\b",
    re.IGNORECASE
)

EXAM_PATTERN = re.compile(
    r"\bEND\s+SEMESTER\s+EXAMINATION\s+QUESTION\s+PAPER\s+PATTERN\b",
    re.IGNORECASE
)

CREDITS_PATTERN = re.compile(
    r"L\s+T\s+P\s+EL\s+Credits\s+Total\s+Marks\s*"
    r"[\r\n\s]+"
    r"\d+\s+\d+\s+\d+\s+\d+\s+(\d+)\s+\d+",
    re.IGNORECASE
)


# =========================================================
# COURSE DETECTION
# =========================================================

def detect_course(text):
    """
    Detect course code and course name.
    """

    match = COURSE_PATTERN.search(text)

    if not match:
        return None, None

    course_code = match.group(1).strip()
    course_name = match.group(2).strip()

    return course_code, course_name


def detect_credits(text):
    """
    Detect the credit value from the
    L T P EL Credits Total Marks row.
    """

    match = CREDITS_PATTERN.search(text)

    if match:
        return match.group(1).strip()

    return None


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):
    """
    Normalize text extracted from the PDF.
    """

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(
        r"[ \t]+$",
        "",
        text,
        flags=re.MULTILINE
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    text = re.sub(
        r"[ \t]{2,}",
        " ",
        text
    )

    return text.strip()


def remove_repeated_headers(text):
    """
    Remove repeated university/document headers.
    """

    patterns = [
        r"SATHYABAMA\s+INSTITUTE\s+OF\s+SCIENCE\s+AND\s+TECHNOLOGY",
        r"SCHOOL\s+OF\s+COMPUTING",
        r"B\.E\s+CSE\s*[–-]\s*DATA\s+SCIENCE",
        r"REGULATIONS\s+2023",
    ]

    for pattern in patterns:

        text = re.sub(
            pattern,
            "",
            text,
            flags=re.IGNORECASE
        )

    # Remove isolated page numbers.
    text = re.sub(
        r"(?m)^\s*\d+\s*$",
        "",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# =========================================================
# SECTION DETECTION
# =========================================================

def find_sections(text):
    """
    Detect major academic sections on a page.

    Returns:
        [
            (section_type, unit_name, start_position)
        ]
    """

    sections = []

    # -------------------------
    # Units
    # -------------------------

    for match in UNIT_PATTERN.finditer(text):

        sections.append(
            (
                "UNIT",
                f"UNIT {match.group(1)}",
                match.start()
            )
        )

    # -------------------------
    # Course Outcomes
    # -------------------------

    for match in COURSE_OUTCOMES_PATTERN.finditer(text):

        sections.append(
            (
                "COURSE OUTCOMES",
                None,
                match.start()
            )
        )

    # -------------------------
    # References
    # -------------------------

    for match in REFERENCES_PATTERN.finditer(text):

        sections.append(
            (
                "REFERENCES",
                None,
                match.start()
            )
        )

    # -------------------------
    # Examination Pattern
    # -------------------------

    for match in EXAM_PATTERN.finditer(text):

        sections.append(
            (
                "EXAMINATION PATTERN",
                None,
                match.start()
            )
        )

    sections.sort(
        key=lambda item: item[2]
    )

    return sections


# =========================================================
# CONTEXT HEADER
# =========================================================

def build_context_header(
    course_code=None,
    course_name=None,
    credits=None,
    section=None,
    unit=None
):
    """
    Build concise structured context.
    """

    context = []

    if course_code:
        context.append(
            f"Course Code: {course_code}"
        )

    if course_name:
        context.append(
            f"Course Name: {course_name}"
        )

    if credits:
        context.append(
            f"Credits: {credits}"
        )

    if section:
        context.append(
            f"Section: {section}"
        )

    if unit:
        context.append(
            f"Unit: {unit}"
        )

    return "\n".join(context)


def enrich_chunk(
    chunk,
    course_code=None,
    course_name=None,
    credits=None,
    section=None,
    unit=None
):
    """
    Add structured metadata/context to the text
    that will later be embedded.
    """

    header = build_context_header(
        course_code=course_code,
        course_name=course_name,
        credits=credits,
        section=section,
        unit=unit
    )

    if header:

        return (
            f"{header}\n\n"
            f"{chunk}"
        )

    return chunk


# =========================================================
# CHUNKING
# =========================================================

def create_context_aware_chunks(pages):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=300,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = []

    # -----------------------------------------------------
    # Persistent course/section state
    # -----------------------------------------------------

    current_course_code = None
    current_course_name = None
    current_credits = None

    current_section = None
    current_unit = None

    # -----------------------------------------------------
    # Process every PDF page
    # -----------------------------------------------------

    for page in pages:

        original_text = clean_text(
            page["text"]
        )

        # -------------------------------------------------
        # Detect a new course BEFORE removing headers.
        # -------------------------------------------------

        course_code, course_name = detect_course(
            original_text
        )

        if course_code:

            current_course_code = course_code
            current_course_name = course_name

            detected_credits = detect_credits(
                original_text
            )

            if detected_credits:
                current_credits = detected_credits

            # New course means previous section context
            # must not carry into the new course.
            current_section = None
            current_unit = None

        # -------------------------------------------------
        # Clean repeated PDF headers
        # -------------------------------------------------

        text = remove_repeated_headers(
            original_text
        )

        if not text:
            continue

        # -------------------------------------------------
        # Find sections on this page
        # -------------------------------------------------

        sections = find_sections(text)

        # =================================================
        # CASE 1:
        # No section heading on this page
        # =================================================

        if not sections:

            page_chunks = text_splitter.split_text(
                text
            )

            for chunk in page_chunks:

                # -----------------------------------------
                # If we have an active course and section,
                # this is a continuation page.
                # -----------------------------------------

                if current_course_code:

                    section = current_section

                    unit = current_unit

                    # If there is no known section yet,
                    # treat it as course information.
                    if section is None:

                        section = (
                            "COURSE INFORMATION"
                        )

                    enriched_text = enrich_chunk(
                        chunk,
                        course_code=current_course_code,
                        course_name=current_course_name,
                        credits=current_credits,
                        section=section,
                        unit=unit
                    )

                    chunks.append({
                        "source": page["source"],
                        "page": page["page"],
                        "course_code": current_course_code,
                        "course_name": current_course_name,
                        "credits": current_credits,
                        "section": section,
                        "unit": unit,
                        "text": enriched_text
                    })

                else:

                    # -------------------------------------
                    # No course detected yet.
                    # This is document-level information.
                    # -------------------------------------

                    enriched_text = enrich_chunk(
                        chunk,
                        section="DOCUMENT INFORMATION"
                    )

                    chunks.append({
                        "source": page["source"],
                        "page": page["page"],
                        "course_code": None,
                        "course_name": None,
                        "credits": None,
                        "section": "DOCUMENT INFORMATION",
                        "unit": None,
                        "text": enriched_text
                    })

            continue

        # =================================================
        # CASE 2:
        # Page contains one or more sections
        # =================================================

        first_section_start = sections[0][2]

        # -------------------------------------------------
        # Content before the first section
        # -------------------------------------------------

        pre_section_text = text[
            :first_section_start
        ].strip()

        if pre_section_text:

            pre_chunks = text_splitter.split_text(
                pre_section_text
            )

            for chunk in pre_chunks:

                if current_course_code:

                    # If this is a course page, this is
                    # course-level information.
                    section = (
                        "COURSE INFORMATION"
                    )

                    enriched_text = enrich_chunk(
                        chunk,
                        course_code=current_course_code,
                        course_name=current_course_name,
                        credits=current_credits,
                        section=section
                    )

                    chunks.append({
                        "source": page["source"],
                        "page": page["page"],
                        "course_code": current_course_code,
                        "course_name": current_course_name,
                        "credits": current_credits,
                        "section": section,
                        "unit": None,
                        "text": enriched_text
                    })

                else:

                    # -------------------------------------
                    # Document introduction
                    # -------------------------------------

                    enriched_text = enrich_chunk(
                        chunk,
                        section="DOCUMENT INFORMATION"
                    )

                    chunks.append({
                        "source": page["source"],
                        "page": page["page"],
                        "course_code": None,
                        "course_name": None,
                        "credits": None,
                        "section": "DOCUMENT INFORMATION",
                        "unit": None,
                        "text": enriched_text
                    })

        # -------------------------------------------------
        # Process every section found on the page
        # -------------------------------------------------

        for i, section_info in enumerate(sections):

            section_type, unit_name, start = (
                section_info
            )

            # Determine where this section ends.
            if i + 1 < len(sections):

                end = sections[i + 1][2]

            else:

                end = len(text)

            section_text = text[
                start:end
            ].strip()

            if not section_text:
                continue

            # ---------------------------------------------
            # Update persistent section state.
            #
            # This is important for continuation pages.
            # ---------------------------------------------

            current_section = section_type
            current_unit = unit_name

            section_chunks = text_splitter.split_text(
                section_text
            )

            for chunk in section_chunks:

                enriched_text = enrich_chunk(
                    chunk,
                    course_code=current_course_code,
                    course_name=current_course_name,
                    credits=current_credits,
                    section=section_type,
                    unit=unit_name
                )

                chunks.append({
                    "source": page["source"],
                    "page": page["page"],
                    "course_code": current_course_code,
                    "course_name": current_course_name,
                    "credits": current_credits,
                    "section": section_type,
                    "unit": unit_name,
                    "text": enriched_text
                })

    return chunks


# =========================================================
# TEST / DEBUG
# =========================================================

if __name__ == "__main__":

    pages = load_all_pdfs()

    chunks = create_context_aware_chunks(
        pages
    )

    print("\n==============================")
    print("SECTION-AWARE CHUNKING")
    print("==============================")

    print(
        "Total pages:",
        len(pages)
    )

    print(
        "Total chunks:",
        len(chunks)
    )

    for i, chunk in enumerate(
        chunks[:15],
        start=1
    ):

        print(
            "\n-----------------------------"
        )

        print(
            "Chunk:",
            i
        )

        print(
            "Source:",
            chunk["source"]
        )

        print(
            "Page:",
            chunk["page"]
        )

        print(
            "Course Code:",
            chunk["course_code"]
        )

        print(
            "Course Name:",
            chunk["course_name"]
        )

        print(
            "Credits:",
            chunk["credits"]
        )

        print(
            "Section:",
            chunk["section"]
        )

        print(
            "Unit:",
            chunk["unit"]
        )

        print(
            "-----------------------------"
        )

        print(
            chunk["text"]
        )