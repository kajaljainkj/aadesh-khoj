# Aadesh Khoj (आदेश खोज): multilingual RAG for Hindi government orders

Ask questions about Indian government orders in **Hindi, Hinglish or English** and get answers that
cite the **order number and date**. It works on the PDFs that break normal RAG pipelines:

- **Legacy-font PDFs (Kruti Dev / DevLys).** Many state-government orders are still typed in
  Kruti Dev, which isn't Unicode. Extract the text and you get `jkT; 'kklu }kjk` instead of
  `राज्य शासन द्वारा`. Embedding that garbage makes the document unsearchable.
- **Scanned orders** with no text layer at all.
- **Broken Unicode PDFs** whose glyph maps produce `का͛नύ क` or doubled matras (`स्कूूल`).

Everything runs locally and costs nothing: Ollama for the LLM, e5 for embeddings, Postgres +
pgvector, and Tesseract. Optionally use Groq's free API for faster answers, with automatic
fallback to local Ollama when the key is missing or the free quota runs out.

## Why this project

| Already exists | Gap this fills |
|---|---|
| Kruti Dev → Unicode converters ([examples](https://github.com/topics/kruti-dev-to-unicode)); [lipi](https://github.com/aparsoft/lipi) decodes legacy-font PDFs | None of these do retrieval or QA |
| Government-scheme chatbots (e.g. Jugalbandi) | Built on clean Unicode content, not legacy-font or scanned orders |
| Generic multilingual RAG tutorials | No measurement of what each ingestion stage actually buys you |

## Architecture

```
PDF ─► PyMuPDF spans ─┬─ font is Kruti Dev/DevLys? ──► convert only those spans ─┐
                      ├─ no text / garbled (>5%)? ───► Tesseract OCR (eng+hin) ──┤
                      └─ normal Unicode ─────────────────────────────────────────┤
                                                                                 ▼
                         normalize (NFC, doubled matras, NUL bytes) ─► metadata (order no, date, dept, subject)
                                                                                 ▼
                         chunk + context header ─► e5 embeddings + Postgres tsvector

question ─► detect hi / hinglish / en ─► [rewrite to formal Hindi via LLM]
          ─► vector search + keyword search ─► RRF fusion (top 20)
          ─► [cross-encoder reranker → top 5] ─► LLM answers in the user's language, citing [n]
```

Design decisions worth talking about in an interview:

- **Span-level conversion, not page-level.** In real legacy documents, dates and numbers are
  often typed in an English font, because in Kruti Dev `/` renders as `ध` and `:` as `रू`.
  Converting the whole page would corrupt `15/03/2024` into `15ध03ध2024`. PyMuPDF exposes the
  font of each span, so only Kruti Dev spans get converted. When font info is missing, the
  text-mode fallback protects date-like tokens.
- **OCR language order matters.** Tesseract with `hin+eng` misread the Latin digits in dates and
  order numbers (`10/06` → `0/06`, `12/04` → `42/04`). With `eng+hin` at 400 DPI the samples
  read 23/23 numbers correctly (100% text similarity) vs 12/23 with `hin+eng`.
- **Query rewrite into formal Hindi.** The corpus is Hindi, while users say "DA", "chhutti",
  "transfer". The LLM maps those to the vocabulary orders actually use (महंगाई भत्ता, अवकाश,
  स्थानांतरण) before searching.
- **Hybrid search.** Order numbers, dates and exact terms are keyword problems. Postgres
  full-text (`simple` config, which tokenizes Devanagari correctly) is fused with vector results
  via Reciprocal Rank Fusion.
- **Two-stage retrieval (optional reranker).** The e5 bi-encoder embeds question and chunk
  separately: fast but approximate. A cross-encoder (`bge-reranker-v2-m3`, multilingual) reads
  question + chunk together and re-scores the top 20, keeping the best 5. It judges against the
  user's original question, not the LLM rewrite. Toggle with `RERANK_ENABLED` or the UI checkbox;
  eval config **E** measures its effect (D vs E use the same rewrite, so only reranking differs).
- **Context headers.** Every chunk is embedded with `subject | क्रमांक | दिनांक` prepended, so a
  chunk from page 2 still knows which order it belongs to.

## Measured on the sample corpus

`data/samples/` has 12 **fictional** orders for a made-up state (5 Kruti Dev, 4 Unicode,
3 scanned). Text similarity to the ground truth after extraction:

| PDF type | Raw extraction | This pipeline | Method |
|---|---|---|---|
| Kruti Dev (5) | 5–11% | 99.4–100% | `krutidev-font` |
| Scanned (3) | 0% (no text) | ~100% | `ocr` |
| Unicode (4) | 96–99.7% | 99.8–100% | `unicode` + normalization |

### Real 2026 orders (MP General Administration Department)

7 real orders downloaded from the department's publications (May–Aug 2026). What the PDFs
actually contained:

| What extraction found | Files | Handled by |
|---|---|---|
| Scanned page with an **invisible English-only OCR layer**: Hindi came out as Latin junk (`ar{rIeer ilrg;T TrIIrdI`) | 5 of 7 | junk-layer detection → re-OCR with `eng+hin` |
| Text layer with broken CID-font mapping | 1 (20 of 21 pages) | garbled-text detection → OCR |
| Pure image, no text layer | 1 | OCR |

None of the 7 had a usable text layer, so a naive pipeline would index pure noise for every
document. The junk OCR layer is the sneaky one: it's valid ASCII, so it passes Unicode checks.
It's detected by (no Devanagari) + (almost no English stopwords) + (a full-page image).
Questions for these orders: `eval/questions_real.jsonl` (24, with reference answers).

**Retrieval results on the real orders** (24 questions, Recall@5 / MRR; the reranker config E
is implemented but not yet measured on this set):

| Config | en R@5 | hi R@5 | hinglish R@5 | all R@5 | all MRR |
|---|---|---|---|---|---|
| A. naive (raw text) | 0.75 | 0.50 | 0.75 | 0.67 | 0.59 |
| B. + junk-layer fix & OCR | 0.88 | 1.00 | 0.88 | 0.92 | 0.90 |
| C. + hybrid search | 1.00 | 1.00 | 0.88 | 0.96 | 0.89 |
| D. + query rewrite (Groq gpt-oss-120b) | 1.00 | 1.00 | 1.00 | **1.00** | **0.97** |

The eval also surfaced a failure: for *"What is the theme of International Yoga Day 2026?"* the
rewrite model **answered instead of rewriting**, inventing a theme ("Yoga for a Sustainable
Future"; the order says "Yoga for Healthy Ageing"). Retrieval still worked, but a hallucinated
query can steer search wrong. Fix: `_rewrite_is_safe()` rejects rewrites that add English words or
numbers not in the question, quotes, or run long, and falls back to the original question; the
original question is also always searched and fused, so one bad rewrite can't hide the right
order.

```bash
python -m app.ingest data/raw --collection real
python -m eval.run_eval --docs data/raw --questions eval/questions_real.jsonl
```

Known limits seen on real data: OCR sometimes misreads digits in dates (e.g. 2026 → 2016), and
order-number / date extraction is less reliable than on the clean samples.

### Sample corpus eval

36 questions (12 fictional orders × en / hi / hinglish), configs A–C:

| Config | en R@5 | hi R@5 | hinglish R@5 | all R@5 | all MRR |
|---|---|---|---|---|---|
| A. naive (raw text) | 0.75 | 0.58 | 0.75 | 0.69 | 0.53 |
| B. + font fix & OCR | 1.00 | 1.00 | 0.92 | 0.97 | 0.97 |
| C. + hybrid search | 1.00 | 1.00 | 0.92 | 0.97 | 0.96 |

With naive extraction, the 8 Kruti Dev and scanned orders are effectively invisible.
Reproduce with `uv run python -m eval.run_eval` (add `--no-llm` to skip the LLM configs D/E).

**Honest caveat:** both eval sets are small (12 and 7 documents), so near-perfect scores mostly
show the ingestion fixes work; a larger set (50+ orders, 100+ questions) is the next step.

## Setup (Windows, with uv)

**Prerequisites:** [uv](https://docs.astral.sh/uv/), [Docker Desktop](https://www.docker.com/products/docker-desktop/),
[Ollama](https://ollama.com), and Tesseract.

**Tesseract (for scanned PDFs):** install the
[UB Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki). During setup, open
*Additional language data* and tick **Hindi**. If `tesseract --version` doesn't work in a new
terminal, set `TESSERACT_CMD` in `.env`. Without Tesseract everything else still works; scanned
pages are just skipped with a warning.

```powershell
cd aadesh-khoj
uv sync                                # creates .venv, installs everything (PyTorch is big)
copy .env.example .env                 # defaults: Ollama + local DB, no keys needed
ollama pull gemma3:4b                  # one time, ~3 GB
docker compose up -d                   # Postgres + pgvector on port 55432

uv run python -m app.ingest data/samples   # first run downloads the e5 model (~1 GB)
uv run python ui.py                        # http://localhost:7860
```

API: `uv run uvicorn app.main:app --reload` → http://localhost:8000/docs

| Endpoint | What it does |
|---|---|
| `POST /ask` | `{"question": "DA kitna badha?", "strategy": "rewrite", "mode": "hybrid"}` |
| `GET /documents` | indexed orders with order no, date, extraction method |
| `POST /documents` | upload a PDF; it is converted / OCR'd and indexed |

Tests: `uv run pytest -q`. Integration tests use the Docker DB if it's running and skip otherwise.
The OCR test needs Tesseract with Hindi.

## LLM providers and fallback

```
LLM_PROVIDER=groq        # ollama | groq | gemini | deepseek | grok
GROQ_API_KEY=gsk_...     # free, no card: https://console.groq.com/keys
LLM_FALLBACK=ollama      # used when the primary has no key, a bad key, hits a rate limit / daily
                         # quota, or has a network/server error
```

After a rate-limit or bad-key error the primary is skipped for `LLM_FALLBACK_COOLDOWN` seconds
(default 60), so users don't wait for a failing call on every question. The UI shows which LLM
actually answered. A `400 Bad Request` is *not* retried on the fallback, because that's a bug,
not an outage.

## Use real orders

Download Hindi orders from your state's department websites (General Administration, Finance and
School Education departments publish many), put them in `data/raw/`, and run:

```powershell
uv run python -m app.ingest data/raw --collection real
```

Set `COLLECTION=real` in `.env` to query them in the UI. The ingest log shows the extraction
method per file, so you can see how many were Kruti Dev or scanned. That stat is a good README
line too. Write 20–30 questions for them in the same JSONL format and run
`python -m eval.run_eval --docs data/raw --questions your_questions.jsonl`.

## Project layout

```
app/
  legacy_fonts.py     Kruti Dev <-> Unicode conversion + detection
  extract.py          per-page method choice: font-aware conversion / OCR / unicode
  textnorm.py         Devanagari normalization, garbled-text detection
  metadata.py         order no, date, department, subject (handles Devanagari digits)
  chunking.py         paragraph/sentence-aware chunks (Hindi '।' aware)
  embeddings.py       local multilingual e5
  db.py               documents + chunks, pgvector, tsvector, RRF
  rag.py              rewrite → hybrid retrieve → [rerank] → grounded answer
  rerank.py           cross-encoder second-stage reranking
  llm.py              one OpenAI-compatible client: Ollama / Groq / Gemini / DeepSeek / Grok, with fallback
  lang.py             hi / hinglish / en detection
  ingest.py, main.py  CLI ingestion, FastAPI
ui.py                 Gradio UI (ask + documents/upload tabs)
scripts/              fictional sample orders + PDF generator
eval/                 36 questions, 4-config ablation
tests/                65 tests (converter, extraction, OCR, DB integration, API, LLM fallback, reranker)
```

## Limitations and next steps

- Only Kruti Dev / DevLys encodings. Chanakya and others need their own tables; the
  per-span font detection makes adding them easy.
- The sample corpus is synthetic. Real orders have tables, stamps and multi-column layouts.
- Next ideas: table extraction, detecting when a newer order
  supersedes an older one, and LLM-as-judge for answer faithfulness.

## Disclaimer

The sample orders are fictional and exist only for testing. The assistant gives information, not
legal advice. Always check the latest official order.
