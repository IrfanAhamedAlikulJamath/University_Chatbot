import re
import chromadb

from embeddings import generate_query_embedding


# =========================================================
# CONFIGURATION
# =========================================================

CHROMA_PATH = "data/chroma"

COLLECTION_NAME = "university_documents"

VECTOR_RESULTS = 10


# =========================================================
# CONNECT TO CHROMADB
# =========================================================

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_collection(
    name=COLLECTION_NAME
)


# =========================================================
# QUERY INTENT DETECTION
# =========================================================

def detect_query_intent(query):
    """
    Identify what type of university information
    the student is asking for.
    """

    query_lower = query.lower()

    # -----------------------------------------------------
    # Credits / course information
    # -----------------------------------------------------

    if any(
        keyword in query_lower
        for keyword in [
            "credit",
            "credits",
            "course code",
            "course name",
            "course objective",
            "course objectives",
            "maximum marks",
            "total marks"
        ]
    ):
        return "COURSE INFORMATION"


    # -----------------------------------------------------
    # Course outcomes
    # -----------------------------------------------------

    if any(
        keyword in query_lower
        for keyword in [
            "course outcome",
            "course outcomes",
            "co",
            "outcomes of"
        ]
    ):
        return "COURSE OUTCOMES"


    # -----------------------------------------------------
    # References
    # -----------------------------------------------------

    if any(
        keyword in query_lower
        for keyword in [
            "reference",
            "references",
            "reference book",
            "reference books",
            "text book",
            "textbooks",
            "books"
        ]
    ):
        return "REFERENCES"


    # -----------------------------------------------------
    # Examination pattern
    # -----------------------------------------------------

    if any(
        keyword in query_lower
        for keyword in [
            "exam pattern",
            "examination pattern",
            "question paper pattern",
            "marks pattern",
            "exam duration",
            "part a",
            "part b"
        ]
    ):
        return "EXAMINATION PATTERN"


    # -----------------------------------------------------
    # Specific unit
    # -----------------------------------------------------

    unit_match = re.search(
        r"\bunit\s*[-]?\s*([1-5])\b",
        query_lower
    )

    if unit_match:
        return f"UNIT {unit_match.group(1)}"


    # -----------------------------------------------------
    # No specific intent detected
    # -----------------------------------------------------

    return None


# =========================================================
# COURSE DETECTION
# =========================================================

def detect_course(query, results):
    """
    Try to determine which course the student is asking about.

    We use the retrieved metadata to identify the course.
    """

    query_lower = query.lower()

    metadatas = results["metadatas"][0]

    # -----------------------------------------------------
    # First try exact course-name matching
    # -----------------------------------------------------

    for metadata in metadatas:

        course_name = metadata.get(
            "course_name",
            ""
        )

        if course_name:

            normalized_name = course_name.lower()

            if normalized_name in query_lower:

                return course_name


    # -----------------------------------------------------
    # Try course-code matching
    # -----------------------------------------------------

    for metadata in metadatas:

        course_code = metadata.get(
            "course_code",
            ""
        )

        if course_code:

            if course_code.lower() in query_lower:

                return metadata.get(
                    "course_name",
                    ""
                )


    # -----------------------------------------------------
    # If no course explicitly appears in the question,
    # don't force a course.
    # -----------------------------------------------------

    return None


# =========================================================
# SCORE RESULTS
# =========================================================

def rank_results(
    query,
    results
):
    """
    Re-rank vector-search results using metadata.

    Vector similarity remains the primary signal.
    Metadata is used as an additional relevance signal.
    """

    documents = results["documents"][0]

    metadatas = results["metadatas"][0]

    distances = results["distances"][0]


    intent = detect_query_intent(
        query
    )

    course_name = detect_course(
        query,
        results
    )


    ranked_results = []


    for i in range(
        len(documents)
    ):

        metadata = metadatas[i]

        distance = distances[i]

        # -------------------------------------------------
        # Start with vector similarity.
        #
        # Smaller distance = more similar.
        # -------------------------------------------------

        score = -distance


        result_course = metadata.get(
            "course_name",
            ""
        )

        result_section = metadata.get(
            "section",
            ""
        )

        result_unit = metadata.get(
            "unit",
            ""
        )


        # -------------------------------------------------
        # COURSE MATCH
        # -------------------------------------------------

        if course_name:

            if (
                result_course.lower()
                == course_name.lower()
            ):

                score += 1.0


        # -------------------------------------------------
        # SECTION MATCH
        # -------------------------------------------------

        if intent:

            if intent == "COURSE INFORMATION":

                if (
                    result_section
                    == "COURSE INFORMATION"
                ):

                    score += 2.0


            elif intent == "COURSE OUTCOMES":

                if (
                    result_section
                    == "COURSE OUTCOMES"
                ):

                    score += 2.0


            elif intent == "REFERENCES":

                if (
                    result_section
                    == "REFERENCES"
                ):

                    score += 2.0


            elif intent == "EXAMINATION PATTERN":

                if (
                    result_section
                    == "EXAMINATION PATTERN"
                ):

                    score += 2.0


            elif intent.startswith("UNIT"):

                if (
                    result_unit
                    == intent
                ):

                    score += 2.0


        ranked_results.append({
            "document": documents[i],
            "metadata": metadata,
            "distance": distance,
            "score": score
        })


    # -----------------------------------------------------
    # Sort by final score.
    # -----------------------------------------------------

    ranked_results.sort(
        key=lambda item: item["score"],
        reverse=True
    )


    return ranked_results


# =========================================================
# SEARCH
# =========================================================

def search_documents(
    query,
    number_of_results=5
):

    # -----------------------------------------------------
    # 1. Convert query to embedding
    # -----------------------------------------------------

    query_embedding = generate_query_embedding(
        query
    )


    # -----------------------------------------------------
    # 2. Retrieve MORE candidates than we finally show.
    #
    # Example:
    #
    # Need 5 results
    #       ↓
    # retrieve 10
    #       ↓
    # rerank
    #       ↓
    # return best 5
    # -----------------------------------------------------

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],

        n_results=VECTOR_RESULTS,

        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )


    # -----------------------------------------------------
    # 3. Re-rank candidates
    # -----------------------------------------------------

    ranked_results = rank_results(
        query,
        results
    )


    # -----------------------------------------------------
    # 4. Return only requested number
    # -----------------------------------------------------

    return ranked_results[
        :number_of_results
    ]


# =========================================================
# DISPLAY RESULTS
# =========================================================

def display_results(
    query,
    results
):

    print("\n==============================")

    print(
        "RETRIEVAL RESULTS"
    )

    print(
        "=============================="
    )

    print(
        "Query:",
        query
    )


    intent = detect_query_intent(
        query
    )

    print(
        "Detected Intent:",
        intent
    )


    for i, result in enumerate(
        results,
        start=1
    ):

        metadata = result[
            "metadata"
        ]

        document = result[
            "document"
        ]

        distance = result[
            "distance"
        ]

        score = result[
            "score"
        ]


        print(
            "\n-----------------------------"
        )

        print(
            f"Result {i}"
        )

        print(
            "-----------------------------"
        )

        print(
            "Source:",
            metadata.get(
                "source",
                ""
            )
        )

        print(
            "Page:",
            metadata.get(
                "page",
                ""
            )
        )

        print(
            "Course Code:",
            metadata.get(
                "course_code",
                ""
            )
        )

        print(
            "Course Name:",
            metadata.get(
                "course_name",
                ""
            )
        )

        print(
            "Credits:",
            metadata.get(
                "credits",
                ""
            )
        )

        print(
            "Section:",
            metadata.get(
                "section",
                ""
            )
        )

        print(
            "Unit:",
            metadata.get(
                "unit",
                ""
            )
        )

        print(
            "Vector Distance:",
            distance
        )

        print(
            "Final Ranking Score:",
            score
        )

        print(
            "\nText:"
        )

        print(
            document
        )


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    query = input(
        "\nEnter your question: "
    )

    results = search_documents(
        query
    )

    display_results(
        query,
        results
    )