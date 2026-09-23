<div align="center">

# 🎓 University Assistance AI Chatbot

### 🤖 RAG-Powered Academic Assistant using Local LLMs

<p>
  An AI-powered university chatbot that retrieves information from academic documents
  and generates grounded answers using Retrieval-Augmented Generation.
</p>

<br>

<img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white">
<img src="https://img.shields.io/badge/LangChain-RAG-1C3C3C?style=for-the-badge">
<img src="https://img.shields.io/badge/Ollama-Local%20LLM-black?style=for-the-badge">
<img src="https://img.shields.io/badge/Gemma%203-4B-8E75B2?style=for-the-badge">
<img src="https://img.shields.io/badge/ChromaDB-Vector%20Database-FF6F61?style=for-the-badge">
<img src="https://img.shields.io/badge/RAG-Retrieval%20Augmented%20Generation-00A67E?style=for-the-badge">

</div>

---

## 🌟 Overview

The **University Assistance AI Chatbot** is a Retrieval-Augmented Generation (RAG) application designed to help students interact with university academic documents using natural language.

Instead of relying only on an LLM's pretrained knowledge, the chatbot first retrieves relevant information from university documents and provides that information as context to a locally running **Gemma 3 4B** model.

The system can answer questions related to:

- 📚 Course information
- 🔢 Course credits
- 🧾 Course codes
- 📖 Course outcomes
- 📑 Units and syllabus information
- 📝 Examination patterns
- 📚 Reference books
- 🎓 University academic regulations

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 📄 PDF Knowledge Base | Uses university academic documents as the source of knowledge |
| 🔍 Semantic Search | Finds relevant information based on meaning |
| 🧠 RAG Pipeline | Combines retrieval with LLM-based answer generation |
| 🤖 Local LLM | Uses Gemma 3 4B through Ollama |
| 🗃️ Vector Database | Stores embeddings using ChromaDB |
| 🎯 Context-Aware Chunking | Preserves course and academic context |
| 📚 Course-Aware Retrieval | Uses course and section metadata during ranking |
| 💬 Continuous Chat | Supports multiple questions in one session |
| 🔒 Local Processing | LLM inference runs locally through Ollama |

---

# 🏗️ Architecture

## 📥 Document Ingestion Pipeline

```text
                📄 UNIVERSITY PDF
                       │
                       ▼
              ┌─────────────────┐
              │   PDF Loader    │
              │     PyPDF       │
              └────────┬────────┘
                       │
                       ▼
             ┌─────────────────────┐
             │  Context-Aware      │
             │  Text Chunking      │
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │ Sentence Transformer│
             │  all-MiniLM-L6-v2   │
             └──────────┬──────────┘
                        │
                        ▼
              ┌─────────────────┐
              │    ChromaDB     │
              │  Vector Store   │
              └─────────────────┘
```

## 📤 Question Answering Pipeline

```text
              👨‍🎓 STUDENT QUESTION
                       │
                       ▼
              ┌─────────────────┐
              │ Query Embedding │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Semantic Search │
              │   + Ranking     │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Relevant PDF    │
              │    Context      │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Prompt + Context│
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   Gemma 3 4B    │
              │   Local LLM     │
              └────────┬────────┘
                       │
                       ▼
                  💬 FINAL ANSWER
```

---

# 🔄 How It Works

### 1️⃣ Document Ingestion

University PDF documents are loaded and their text is extracted page by page.

### 2️⃣ Context-Aware Chunking

The extracted content is divided into meaningful chunks while preserving important academic information such as:

```text
Course Code
Course Name
Credits
Section
Unit
Page Number
Source Document
```

This helps prevent important course information from becoming disconnected from the content it belongs to.

### 3️⃣ Embedding Generation

Each document chunk is converted into a numerical vector using:

```text
Sentence Transformers
        ↓
all-MiniLM-L6-v2
```

The same embedding model is used for both document chunks and student queries.

### 4️⃣ Vector Storage

The generated embeddings, document text, and metadata are stored in **ChromaDB**.

```text
Document Chunk
      +
Embedding
      +
Metadata
      ↓
   ChromaDB
```

### 5️⃣ Query Retrieval

When a student asks a question:

```text
Question
   ↓
Query Embedding
   ↓
ChromaDB Search
   ↓
Relevant Chunks
   ↓
Course / Section-aware Ranking
```

### 6️⃣ Grounded Answer Generation

The retrieved university context is passed to **Gemma 3 4B** through a controlled prompt.

The model is instructed to:

- Use the retrieved university context
- Avoid unsupported assumptions
- Avoid mixing information between courses
- Prefer explicitly labelled information
- State when information cannot be found

### 7️⃣ Final Response

Gemma generates a natural-language response based on the retrieved university information.

---

# 🧠 Why RAG?

A general-purpose LLM may not contain institution-specific information such as university course codes, credits, regulations, course outcomes, or examination patterns.

RAG solves this by retrieving relevant information from the university documents before generating the answer.

```text
Student Question
       │
       ▼
Semantic Retrieval
       │
       ▼
University Documents
       │
       ▼
Relevant Context
       │
       ▼
Gemma 3 4B
       │
       ▼
Grounded Response
```

This allows the chatbot to combine:

**Retrieval → Context → Generation**

rather than relying entirely on the LLM's pretrained knowledge.

---

# 🛠️ Technology Stack

| Category | Technology |
|---|---|
| 💻 Programming Language | Python |
| 🧠 Large Language Model | Gemma 3 4B |
| ⚡ LLM Runtime | Ollama |
| 🔗 LLM Framework | LangChain |
| 🔢 Embeddings | Sentence Transformers |
| 🧩 Embedding Model | all-MiniLM-L6-v2 |
| 🗄️ Vector Database | ChromaDB |
| 📄 PDF Processing | PyPDF |
| 🛠️ Development | VS Code |
| 🌐 Version Control | Git & GitHub |

---

# 📂 Project Structure

```text
University_Chatbot/
│
├── 📁 data/
│   ├── 📁 pdfs/
│   │   └── regulations.pdf
│   │
│   └── 📁 chroma/
│
├── 📁 src/
│   ├── pdf_loader.py
│   ├── text_processor.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── llm.py
│   └── rag_pipeline.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

> **Note:** University PDF files and local ChromaDB storage are excluded from Git using `.gitignore`.

---

# 🚀 Getting Started

## Prerequisites

Make sure you have:

- Python 3.x
- Git
- Ollama

---

## 1️⃣ Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd University_Chatbot
```

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

## 3️⃣ Install Gemma 3 4B

```bash
ollama pull gemma3:4b
```

Verify the installation:

```bash
ollama list
```

## 4️⃣ Add University Documents

Place your PDF documents inside:

```text
data/pdfs/
```

Example:

```text
data/pdfs/regulations.pdf
```

## 5️⃣ Build the Vector Store

```bash
python src/vector_store.py
```

This performs:

```text
PDF
 ↓
Text Extraction
 ↓
Context-Aware Chunking
 ↓
Embeddings
 ↓
ChromaDB
```

## 6️⃣ Start the Chatbot

Run the following command from the **project root directory**:

```bash
python src/rag_pipeline.py
```

The chatbot will remain active until:

```text
exit
```

is entered.

---

# 💬 Example

```text
==============================
UNIVERSITY ASSISTANCE CHATBOT
==============================
Type 'exit' to quit.

Student: How many credits does Programming in Java have?

Searching university documents...
Retrieved 5 relevant chunks.

Generating answer...

Chatbot: Programming in Java has 4 credits.

Student: What are the course outcomes?

Searching university documents...
Retrieved 5 relevant chunks.

Generating answer...

Chatbot: ...
```

---

# 🎯 Design Principles

### 🔎 Retrieval First

The system retrieves university-specific information before asking the LLM to generate an answer.

### 🎯 Context-Aware Retrieval

Course and section metadata are preserved during chunking and used during retrieval.

### 🧠 Grounded Generation

Gemma is instructed to use retrieved context instead of relying on unsupported information.

### 🌡️ Deterministic Generation

The LLM uses:

```python
temperature=0
```

to reduce unnecessary randomness for factual academic queries.

### 🔐 Local AI

The LLM runs locally through Ollama, avoiding dependency on a hosted LLM API for inference.

---

# 📊 Current Knowledge Base

The current implementation processes:

```text
📄 University Regulation PDF
        ↓
148 Pages
        ↓
651 Searchable Chunks
        ↓
384-Dimensional Embeddings
        ↓
ChromaDB
```

---

# 🔮 Future Enhancements

- 🧠 Conversation memory and follow-up questions
- 🌐 Web-based chatbot interface
- 📑 Source and page citations
- 📚 Multi-document and multi-department support
- 📊 Retrieval and answer evaluation
- 📋 Improved table-aware PDF parsing
- ⚡ Streaming LLM responses
- 🔐 Authentication and student-specific features

---

# ⚠️ Limitations

The chatbot's performance depends on the quality and structure of the source documents.

Current limitations include:

- Complex PDF tables may lose their original structure during text extraction.
- Poorly formatted documents can affect retrieval quality.
- Ambiguous information may occasionally require additional retrieval or validation.
- The current interface is terminal-based.

---

# 👨‍💻 Author

<div align="center">

## **Irfan Ahamed Alikul Jamath**

**B.E. Computer Science & Engineering — Data Science**

Sathyabama Institute of Science and Technology

<br>

⭐ If you find this project interesting, consider giving the repository a star!

</div>

---

<div align="center">

### 🚀 Built with Python • RAG • ChromaDB • Ollama • Gemma

</div>
