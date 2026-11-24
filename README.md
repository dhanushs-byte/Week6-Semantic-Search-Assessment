# Week 6 Weekly Assessment — Embeddings + Vector Databases

This project implements all four tasks in the supplied Week 6 assessment PDF.

## Technology choices

- Local embeddings: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- Optional API embedding comparison: OpenAI-compatible embeddings endpoint (Task 1 only)
- ANN experiment: FAISS exact inner-product search vs HNSW
- Vector database: Qdrant local persistent mode (no PostgreSQL required)
- API: FastAPI
- Tests: pytest

> Use Python 3.11 or 3.12 recommended. Some ML wheels may not yet be available for Python 3.14 on Windows.

## 1. Setup (Windows PowerShell)

```powershell
cd week6_assessment_complete
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```

The first run downloads `all-MiniLM-L6-v2` from Hugging Face. Internet is required for this first download. After caching, local embedding inference can run offline.

No API key is required for the core project. To run the optional Task 1 API comparison, add your API key and (if needed) base URL in `.env`. Never commit `.env`.

## 2. Project structure

```text
week6_assessment_complete/
├── data/documents/             # Five sample policy/manual documents
├── scripts/
│   ├── task1_embeddings.py
│   ├── task2_chunking.py
│   ├── task3_vector_storage.py
│   └── task4_api_evaluation.py
├── src/
│   ├── embeddings.py
│   ├── chunking.py
│   ├── vector_store.py
│   ├── evaluation.py
│   └── api.py
├── tests/
│   ├── conftest.py
│   └── test_api.py
├── .env.example
├── .gitignore
└── requirements.txt
```

## 3. Run each task

Run from the project root, with `.venv` activated:

```powershell
python scripts/task1_embeddings.py
python scripts/task2_chunking.py
python scripts/task3_vector_storage.py
python scripts/task4_api_evaluation.py
```

Task 1 prints local embedding dimensions/time, optional API comparison, hand-coded cosine checks, similarity matrix, top pairs, three semantic-search queries, duplicate-like pairs, and K-Means clusters.

Task 2 extracts PDF/DOCX/TXT/MD files, compares chunking strategies and sizes, attaches metadata, prints citation strings, and demonstrates Qdrant upserts with stable location-based IDs. Replace the five sample files in `data/documents/` with your own assessment documents when available.

Task 3 runs a reproducible FAISS experiment on 50,000 random 384-dimensional vectors, compares exact and HNSW query time and recall@10, stores chunks in Qdrant, filters, re-upserts and deletes by source.

Task 4 evaluates 15 sample queries for two chunk-size configurations and reports hit@1, hit@3, hit@5 and MRR. Expected sources are explicitly listed in the script. This is a demonstration evaluation set; for a real assessment, write questions whose expected source/page you have manually verified.

## 4. Start the FastAPI service

```powershell
uvicorn src.api:app --reload
```

Open `http://127.0.0.1:8000/docs`.

Endpoints:
- `GET /health`
- `POST /search` — JSON: `{"query":"How do I reset my password?","top_k":5,"doc_type":"txt","min_score":0.2}`
- `POST /documents` — multipart file upload (PDF, DOCX, TXT, MD), extracts, embeds and indexes it.

Protected endpoints require the `x-api-key` header. The default demo key is `week6-demo-key`; set your own `API_KEY` in `.env` before using it beyond local demonstration.

Example PowerShell search:

```powershell
$headers = @{ "x-api-key" = "week6-demo-key" }
$body = @{ query = "How do I reset my password?"; top_k = 5 } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/search" -Method Post -Headers $headers -ContentType "application/json" -Body $body
```

## 5. Run tests

```powershell
pytest -q
```

The six tests cover health, successful search, top_k validation, missing key, invalid key, and document upload/index. Tests use a deterministic lightweight test embedder and in-memory Qdrant; production uses MiniLM.

## 6. Assessment notes and honest interpretation

- The PDF's sample figures (for example, a particular chunk count, milliseconds or retrieval score) are illustrative expected-output examples, not guaranteed results. Actual timings and scores depend on hardware, data and package versions.
- The chunk-size hit-rate demonstration checks whether the expected source/page is represented by at least one chunk. It is a coverage sanity check, not a semantic retrieval score.
- Qdrant local mode creates the collection with HNSW configuration. For a production deployment, use a Qdrant server and inspect its collection/index configuration through the server API.
- The local model produces 384-dimensional vectors. If you change embedding models, recreate the collection with the matching vector dimension and re-embed every document and query with the same model.
- Stable IDs are derived from source/page/section/chunk position. Re-indexing the same location updates the existing point. If chunk boundaries shift substantially after editing a document, downstream chunk positions may change; a content-aware reconciliation strategy may be needed.
- DOCX files do not have stable page boundaries in the standard parser, so they are represented as page 1. PDF pages retain their actual page numbers.
## Retrieval Evaluation

Evaluated 15 queries against the indexed sample documents using two chunk-size configurations.

| Metric | Config A (256) | Config B (512) |
|---|---:|---:|
| Hit@1 | 0.667 | 0.667 |
| Hit@3 | 0.667 | 0.667 |
| Hit@5 | 0.733 | 0.733 |
| MRR | 0.683 | 0.680 |

### Decision

Config A (chunk size 256) achieved a slightly higher MRR of 0.683 compared with 0.680 for Config B (chunk size 512). Both configurations achieved identical Hit@1, Hit@3 and Hit@5 scores.

The difference is small, so additional evaluation on a larger and more diverse document collection is recommended before selecting a production configuration.

### Limitations

- Evaluation used 15 queries.
- Results depend on the sample documents and expected source labels.
- The evaluation scores should not be treated as production-level retrieval benchmarks.