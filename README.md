# Interview Trainer Agent

**AICTE 2026 Project — Problem Statement #22**

An AI-powered interview preparation system that generates personalised mock interviews, evaluates answers, and produces a detailed readiness report — powered by **IBM Granite 4 H Small** via **IBM watsonx.ai** and a **RAG pipeline**.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [AICTE Problem Statement](#aicte-problem-statement)
3. [Features](#features)
4. [Architecture](#architecture)
5. [RAG Pipeline](#rag-pipeline)
6. [Technology Stack](#technology-stack)
7. [Project Structure](#project-structure)
8. [Installation](#installation)
9. [Environment Variables](#environment-variables)
10. [How to Run](#how-to-run)
11. [Usage Workflow](#usage-workflow)
12. [Evaluation Methodology](#evaluation-methodology)
13. [Limitations](#limitations)
14. [Security](#security)
15. [Credits](#credits)

---

## Project Overview

The Interview Trainer Agent is a full-stack AI system that simulates a job interview. Given a candidate's profile, experience level, target role, and optionally their resume, the system:

- Generates personalised interview questions using IBM Granite
- Evaluates each answer with a score (1–10), strengths, weaknesses, and specific improvement advice
- Adapts question difficulty in real time based on candidate performance
- Produces a final readiness report with an overall score and an actionable improvement plan

---

## AICTE Problem Statement

**Problem Statement #22 — Interview Trainer Agent**

> Build an AI system that prepares candidates for job interviews by generating personalised questions based on candidate profile, experience level, target role, resume, interview type, and difficulty. The system should use retrieval-augmented generation (RAG) to ground outputs in interview knowledge, and use IBM Granite via IBM watsonx.ai for all generative tasks.

---

## Features

| Feature | Status |
|---|---|
| IBM Granite 4 H Small integration via watsonx.ai | Complete |
| RAG knowledge base (technical, behavioral, HR, evaluation) | Complete |
| FAISS vector store with sentence-transformer embeddings | Complete |
| PDF resume parsing (text extraction + Granite-based info extraction) | Complete |
| Personalised question generation (role, experience, difficulty, resume) | Complete |
| Answer evaluation (score, strengths, weaknesses, STAR analysis) | Complete |
| Adaptive difficulty (score-based escalation/de-escalation) | Complete |
| Mock interview session (8–10 adaptive questions) | Complete |
| Final readiness report with improvement plan | Complete |
| Multi-page Streamlit UI | Complete |
| Secure credential handling (env vars, never hardcoded) | Complete |

---

## Architecture

```
Candidate Profile + Resume
          |
    Resume Parsing (pypdf + Granite)
          |
   Interview Configuration
   (role / type / difficulty)
          |
   RAG Knowledge Retrieval
   (FAISS + sentence-transformers)
          |
   Relevant Context Chunks
          |
      IBM Granite 4 H Small
      (via watsonx.ai SDK)
          |
  Personalised Interview Question
          |
    Candidate Answer
          |
      IBM Granite Evaluation
          |
   Score + Feedback + STAR Analysis
          |
   Adaptive Follow-up Question
          |
   Final Interview Readiness Report
```

---

## RAG Pipeline

1. **Load** — `rag/document_loader.py` loads all `.txt` files from `data/interview_knowledge/`
2. **Chunk** — `rag/chunker.py` splits documents into 500-word overlapping chunks
3. **Embed** — `rag/embeddings.py` uses `sentence-transformers/all-MiniLM-L6-v2` to generate 384-dim embeddings
4. **Store** — `rag/vector_store.py` builds and persists a FAISS `IndexFlatL2` index
5. **Retrieve** — `rag/retriever.py` takes a query, embeds it, searches the FAISS index, and returns the top-5 most relevant chunks as formatted context
6. **Inject** — The retrieved context is injected into every Granite prompt for grounded generation

Knowledge base files (`data/interview_knowledge/`):
- `technical_concepts.txt` — Python, OOP, data structures, algorithms, databases, APIs, Git
- `role_questions.txt` — Role-specific Q&A for Python Developer, Data Scientist, ML Engineer, etc.
- `behavioral_hr.txt` — STAR method, behavioral competencies, HR questions
- `evaluation_criteria.txt` — Scoring rubrics, evaluation criteria, interviewer expectations

---

## Technology Stack
<img width="1453" height="548" alt="Screenshot 2026-09-08 133748" src="https://github.com/user-attachments/assets/cdb55d39-8cf6-4ed1-bd69-c207cda62c97" />


| Layer | Technology |
|---|---|
| LLM | IBM Granite 4 H Small (`ibm/granite-4-h-small`) |
| LLM Platform | IBM watsonx.ai |
| LLM SDK | `ibm-watsonx-ai` (Python) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector Store | FAISS (`faiss-cpu`) |
| Resume Parsing | `pypdf` |
| Frontend | Streamlit |
| Config | `python-dotenv` |
| Language | Python 3.10+ |

---

## Project Structure

```
interview-trainer-agent/
|
|-- app.py                          # Streamlit application entry point
|-- requirements.txt
|-- .env                            # Your local credentials (never committed)
|-- .env.example                    # Safe template
|-- .gitignore
|-- README.md
|-- test_phase1.py                  # IBM Granite connection test
|
|-- config/
|   |-- settings.py                 # Env var loading + validation
|
|-- agents/
|   |-- interview_agent.py          # Orchestrates the full interview session
|   |-- question_generator.py       # Granite question + model answer generation
|   |-- evaluator.py                # Granite-based answer evaluation
|
|-- rag/
|   |-- document_loader.py          # Load knowledge base .txt files
|   |-- chunker.py                  # Split documents into chunks
|   |-- embeddings.py               # sentence-transformers embedder
|   |-- vector_store.py             # FAISS index build/save/load/search
|   |-- retriever.py                # High-level RAG retriever
|
|-- llm/
|   |-- granite.py                  # IBM watsonx.ai SDK wrapper
|
|-- resume/
|   |-- parser.py                   # PDF parsing + Granite info extraction
|
|-- data/
|   |-- interview_knowledge/        # Knowledge base text files
|   |   |-- technical_concepts.txt
|   |   |-- role_questions.txt
|   |   |-- behavioral_hr.txt
|   |   |-- evaluation_criteria.txt
|   |
|   |-- vector_store/               # Auto-generated FAISS index (not committed)
|
|-- utils/
    |-- helpers.py                  # Score colours, labels, formatting
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/kummariBharath/Interview-Trainer-Agent
cd interview-trainer-agent
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Key packages installed:
- `ibm-watsonx-ai` — IBM watsonx.ai Python SDK
- `sentence-transformers` — local embedding model
- `faiss-cpu` — vector similarity search
- `pypdf` — PDF text extraction
- `streamlit` — web UI
- `python-dotenv` — environment variable loading

---

## Environment Variables

Copy `.env.example` to `.env` and fill in your IBM Cloud credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```
WATSONX_APIKEY=your-ibm-cloud-api-key-here
WATSONX_PROJECT_ID=de532dc4-efe9-49fd-a69b-4f02c40a7403
WATSONX_URL=https://us-south.ml.cloud.ibm.com
WATSONX_MODEL_ID=ibm/granite-4-h-small
```

> **Security:** `.env` is in `.gitignore` and is never committed. The API key is never printed or logged.

---

## How to Run

### Test IBM Granite connection (Phase 1 verification)

```bash
python test_phase1.py
```

Expected output:
```
[1/4] Loading configuration...   [OK]
[2/4] Connecting to IBM watsonx.ai...   [OK]
[3/4] Sending test prompt...
[4/4] Granite response: A Python list is an ordered, mutable collection...
[OK] Phase 1 COMPLETE - IBM Granite connection verified.
```

### Run the Streamlit application

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## Usage Workflow

1. **Home** — Read the overview. Test the IBM Granite connection.
2. **Candidate Profile** — Enter your name, target role, experience level. Optionally upload your resume PDF.
3. **Interview Configuration** — Choose interview type (Technical / HR / Behavioral / Mixed) and difficulty (Easy / Medium / Hard).
4. **Interview Session** — Answer adaptive questions generated by Granite. After each answer:
   - The system evaluates your answer (score, strengths, weaknesses, improvement advice)
   - The next question adapts based on your performance
   - You can request a model answer for any question
5. **Final Report** — View your overall score, category scores, strong/weak areas, topics to revise, and a personalised improvement plan.

---

## Evaluation Methodology

Each answer is evaluated by IBM Granite on eight dimensions:

| Dimension | Weight |
|---|---|
| Relevance | Whether the answer addresses the question |
| Technical Accuracy | Factual correctness |
| Completeness | Coverage of key concepts |
| Clarity | Communication quality and structure |
| Depth | Level of elaboration appropriate to experience |
| Examples | Use of concrete examples |
| Reasoning | Problem-solving and analytical thinking |
| STAR Structure | For behavioral questions only |

**Score scale:**
- 9–10: Exceptional — all key points covered with depth and clarity
- 7–8: Good — most key points with minor gaps
- 5–6: Adequate — basics covered, missing depth
- 3–4: Weak — partial coverage with notable errors
- 1–2: Poor — largely irrelevant or blank

> Note: Scores are AI-based practice evaluations, not objective hiring decisions.

---

## Limitations

- The knowledge base is curated and static. It does not cover all roles or niche domains.
- Resume parsing quality depends on PDF text extractability (scanned/image PDFs are not supported).
- Granite evaluation scores are approximations and should not be used as definitive hiring assessments.
- The vector store is built on startup; very large knowledge bases may be slow to index on first run.
- Internet connectivity is required for all IBM Granite calls.

---

## Security

- Credentials are loaded exclusively from environment variables via `python-dotenv`.
- The API key is never printed, logged, or hardcoded anywhere in the codebase.
- `.env` is in `.gitignore` and must never be committed.
- Error messages scrub the API key value before display.
- `ibm-watsonx-ai` SDK manages IAM token exchange internally; bearer tokens are never exposed.

---

## Credits

- **IBM watsonx.ai** — LLM platform
- **IBM Granite 4 H Small** — generative AI model
- **sentence-transformers** (UKPLab) — `all-MiniLM-L6-v2` embedding model
- **FAISS** (Meta AI) — vector similarity search
- **Streamlit** — web application framework
- **pypdf** — PDF text extraction
- **AICTE 2026** — Problem Statement #22
