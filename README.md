# Discharge Summary Agent

> **Agentic AI system** that reads raw patient source notes and produces structured, clinically safe discharge summary drafts for clinician review.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Part 1 — Discharge Summary Agent](#part-1--discharge-summary-agent)
- [Part 2 — Learning from Doctor Edits](#part-2--learning-from-doctor-edits)
- [Design Decisions and Trade-offs](#design-decisions-and-trade-offs)
- [Limitations](#limitations)

---

## Overview

This system takes a folder of patient PDFs (admission notes, progress notes, lab results, medication records) and produces:

- A **structured discharge summary draft** with all required clinical sections
- A **step-by-step reasoning trace** showing every agent decision
- **Flags and escalations** for conflicts, missing data, and dangerous drug interactions

The output is always a **draft for clinician review** — the agent never auto-finalises a clinical document or fabricates facts.

---

## Architecture

### High-Level System Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        main.py                                   │
│  python main.py --patient patient_1 [--learn] [--iterations N]  │
└──────────────┬──────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────┐
│  Stage 1: PDF Ingestion                                           │
│  ┌─────────────────────────────────────────────┐                 │
│  │  PDFLoader                                   │                 │
│  │  ├── PyMuPDF  → text-based PDFs (instant)   │                 │
│  │  └── pytesseract (OCR) → scanned PDFs       │                 │
│  │       ├── batch_size=3 pages (low RAM)       │                 │
│  │       └── .ocr_cache.json → skip on re-run  │                 │
│  └─────────────────────────────────────────────┘                 │
└──────────────┬───────────────────────────────────────────────────┘
               │ docs {name: DocumentContent}
               ▼
┌──────────────────────────────────────────────────────────────────┐
│  Stage 2: RAG Indexing                                            │
│  ┌─────────────────────────────────────────────┐                 │
│  │  RAGPipeline                                 │                 │
│  │  ├── VectorStore (ChromaDB + FastEmbed)      │                 │
│  │  │    ├── BAAI/bge-small-en-v1.5 (ONNX)     │                 │
│  │  │    ├── Batched embedding (32 chunks/batch)│                 │
│  │  │    └── Persisted → artifacts/vectorstore/ │                 │
│  │  └── BM25 fallback (pure Python)             │                 │
│  └─────────────────────────────────────────────┘                 │
└──────────────┬───────────────────────────────────────────────────┘
               │ pre_indexed_rag (passed directly — no re-OCR)
               ▼
┌──────────────────────────────────────────────────────────────────┐
│  Stage 3: Agent Run  (LangGraph)                                  │
│                                                                    │
│   START → ingest_documents → plan → batch_extract                 │
│              │ (skip if RAG pre-loaded)  │                         │
│              │               └── 1 LLM call, 13 sections          │
│              │                   + correction memory injection     │
│              ▼                                                     │
│   reconcile_medications → check_conflicts → drug_interaction      │
│              │ (parallel)        │ (parallel)        │             │
│              ▼                   ▼                   ▼             │
│          flag changes      flag conflicts      escalate HIGH       │
│              └──────────────────┴───────────────────┘             │
│                                 ▼                                  │
│                         compile_summary → END                      │
│                                                                    │
│   Hard step cap: max_iterations=20 (agent cannot run forever)     │
└──────────────┬───────────────────────────────────────────────────┘
               │ result {summary, trace, escalations}
               ▼
┌──────────────────────────────────────────────────────────────────┐
│  Stage 4: Learning Loop  [--learn flag]  (Part 2)                 │
│                                                                    │
│   for iteration in 1..N:                                           │
│     Agent generates draft                                          │
│     MockDoctor corrects it  (hidden editing policy)               │
│     RewardCalculator scores edit distance                          │
│     CorrectionMemory stores corrections                            │
│     Next iteration: agent sees corrections in prompt              │
│                                                                    │
│   Output: improvement curve + before/after metrics                 │
└──────────────────────────────────────────────────────────────────┘
```

### LangGraph Agent Nodes

```
                    ┌─────────────────┐
              START │ ingest_documents │  Load PDFs + index RAG
                    └────────┬────────┘  (skip if pre-indexed)
                             │
                    ┌────────▼────────┐
                    │      plan       │  Parallel RAG probe per section
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  batch_extract  │  1 LLM call for all 13 sections
                    │                 │  + correction memory injected
                    └────────┬────────┘
                             │
                    ┌────────▼─────────────┐
                    │ reconcile_medications │  Flag added/stopped meds
                    └────────┬─────────────┘
                             │
                    ┌────────▼──────────┐
                    │  check_conflicts  │  Parallel LLM conflict checks
                    └────────┬──────────┘
                             │
                    ┌────────▼────────────────┐
                    │ drug_interaction_check  │  Mock tool + escalation
                    └────────┬────────────────┘
                             │
                    ┌────────▼────────┐
                    │ compile_summary │  Validate + escalate CRITICAL
                    └────────┬────────┘
                             │
                           END
```

### Part 2 — Learning Loop

```
Iteration 1               Iteration 2               Iteration N
──────────                ──────────                ──────────
Agent draft               Agent draft               Agent draft
    │                         │ ▲                       │ ▲
    ▼                         │ │ corrections            │ │
MockDoctor edit           correction memory         correction memory
    │                         │                         │
    ▼                         ▼                         ▼
RewardCalc score          RewardCalc score          RewardCalc score
edit_distance=0.45        edit_distance=0.28        edit_distance=0.12
reward=0.55               reward=0.72               reward=0.88
    │                         │                         │
    └──► CorrectionMemory ────┘                         │
              └────────────────────────────────────────┘
                    artifacts/memory/       (persisted)
                    artifacts/metrics/      (improvement curve)
```

---

## Tech Stack

| Layer | Technology | Reason |
|---|---|---|
| Agent framework | LangGraph | Explicit state machine, conditional edges, step cap |
| LLM | Claude Haiku (Anthropic) | Fast, cheap, strong instruction following |
| Vector store | ChromaDB (persistent) | Local, no server, survives restarts |
| Embeddings | FastEmbed + BAAI/bge-small-en-v1.5 | ONNX-based, ~33MB, no GPU needed |
| BM25 fallback | Pure Python | Zero deps, works if ChromaDB unavailable |
| PDF text extraction | PyMuPDF | Fast native extraction for text PDFs |
| PDF OCR | pytesseract + pdf2image + poppler | Handles scanned image PDFs; batched for low RAM |
| Package manager | uv | Fast, reproducible installs |
| Config | PyYAML + dot-access | `CONFIG.agent.max_iterations` — no magic strings |
| Logging | Python logging + RotatingFileHandler | Mirrors sales-price-main pattern |
| Data validation | Pydantic v2 | Typed discharge summary model |
| Parallelism | ThreadPoolExecutor | Parallel RAG queries + conflict checks |
| Edit distance | difflib (stdlib) | Reward signal for learning loop, no extra deps |

---

## Project Structure

```
discharge-agent/
│
├── main.py                           Entry point — runs all 4 stages
├── config.yaml                       All configuration (dot-access)
├── setup.py                          Package setup
├── pyproject.toml                    Dependencies (uv)
├── .env.example                      API key template
│
├── src/
│   ├── logger/        __init__.py    Rotating file + console log handler
│   ├── exception/     __init__.py    DischargeAgentException(e, sys)
│   ├── constants/     __init__.py    Stage names, sections, drug interaction DB
│   ├── config/        __init__.py    CONFIG = load_config() with dot-access
│   │
│   ├── components/                   Stateless business logic classes
│   │   ├── pdf_loader.py             PyMuPDF + batched OCR + disk cache
│   │   ├── rag_pipeline.py           VectorStore + BM25 fallback
│   │   ├── vector_store.py           ChromaDB + FastEmbed batched embedding
│   │   ├── agent_tools.py            DrugInteractionTool, EscalationTool
│   │   ├── mock_doctor.py            Simulated reviewer (hidden policy)
│   │   ├── reward_calculator.py      Edit distance + section match rate
│   │   └── correction_memory.py      Persisted (draft, corrected) pair store
│   │
│   ├── agent/                        LangGraph agent internals
│   │   ├── state.py                  AgentState TypedDict
│   │   ├── models.py                 DischargeSummary, FlagLevel, etc.
│   │   ├── tracer.py                 Step-by-step observability trace
│   │   ├── nodes.py                  All 7 graph node functions
│   │   └── graph.py                  build_graph() + run_agent()
│   │
│   └── pipeline/                     Stage orchestrators
│       ├── stage01_pdf_ingestion.py
│       ├── stage02_rag_indexing.py
│       ├── stage03_agent_run.py
│       └── stage04_learning_loop.py
│
├── artifacts/
│   ├── patient_1/                    Input PDFs
│   ├── patient_2/
│   │   └── notes.ocr_cache.json      OCR cache (skips re-OCR on re-runs)
│   ├── vectorstore/                  ChromaDB persistent collections
│   ├── outputs/                      Generated summaries and traces
│   │   ├── summary_patient_1.md
│   │   ├── summary_patient_1.json
│   │   └── trace_patient_1.json
│   ├── memory/                       Correction memory (Part 2)
│   └── metrics/                      Improvement curves (Part 2)
│
├── logs/                             Rotating log files (5MB max, 3 backups)
└── tests/
    ├── test_architecture.py          14 unit tests — no API key needed
    └── test_rag_only.py              BM25 + tools logic tests
```

---

## Setup

### System Dependencies

```bash
# macOS
brew install tesseract poppler

# Verify
tesseract --version
which pdftoppm
```

### Python Environment

```bash
# Create virtual environment with uv
uv venv
source .venv/bin/activate       # Mac/Linux
.venv\Scripts\activate          # Windows

# Install all dependencies
uv pip install -e .

# Or with pip
pip install -e .
```

### API Key

```bash
cp .env.example .env
# Open .env and set:
# ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### Add Patient PDFs

```
artifacts/
├── patient_1/
│   ├── admission_note.pdf
│   ├── progress_notes.pdf
│   └── labs.pdf
└── patient_2/
    └── notes.pdf
```

---

## Usage

```bash
# Part 1 — single patient
python main.py --patient patient_1

# Part 1 — all patients in artifacts/
python main.py

# Part 1 + Part 2 learning loop (5 iterations from config)
python main.py --patient patient_1 --learn

# Custom number of learning iterations
python main.py --patient patient_1 --learn --iterations 3

# Use a smarter model for better extraction
DISCHARGE_MODEL=claude-sonnet-4-6 python main.py --patient patient_1

# Run architecture tests (no API key required)
python tests/test_architecture.py
```

### Output Files

```
artifacts/outputs/
├── summary_patient_1.md        Human-readable discharge summary draft
├── summary_patient_1.json      Structured JSON (all fields + flags + escalations)
└── trace_patient_1.json        Full step-by-step agent reasoning trace

artifacts/metrics/              (Part 2 only)
├── metrics_patient_1.json      Full improvement curve data
└── metrics_patient_1.txt       Before/after table (human-readable)

logs/
└── 06_02_2026_14_30_00.log     Rotating log file
```

---

## Part 1 — Discharge Summary Agent

### Hard Safety Guarantees

| Requirement | How it is enforced |
|---|---|
| Never fabricate | LLM prompt enforces MISSING/PENDING/CONFLICT markers; `_apply_value` never fills absent fields |
| No infinite loop | `max_iterations` cap on graph edge condition |
| Pending data | Recorded as `PendingResult`, never inferred or filled in |
| Undocumented med changes | Set-difference reconciliation; all changes without reason flagged |
| Document conflicts | Multi-source LLM check; CRITICAL flag raised — not resolved automatically |
| Tool failures | Try/except with retry on DrugInteractionTool; fallback to partial result |
| Always a draft | `DischargeSummary.is_draft = True` is hardcoded; always printed on output |

### Performance Optimisations

| Optimisation | Impact |
|---|---|
| OCR disk cache | Scanned PDFs: ~10 min → ~0s on re-runs |
| ChromaDB persistence | Re-embedding skipped on re-runs |
| RAG passed Stage 2 to Stage 3 | Eliminates duplicate OCR in Stage 3 |
| 1 batch LLM call for all 13 sections | ~40–60s → ~8–12s |
| Parallel RAG queries (ThreadPoolExecutor) | 13 serial → concurrent |
| Parallel conflict checks | 3 serial LLM calls → concurrent |

---

## Part 2 — Learning from Doctor Edits

### Approach: Correction Memory Injection

Rather than fine-tuning (needs large data, expensive compute), this system uses **in-context learning**: each iteration's doctor corrections are stored and injected as few-shot examples into the next iteration's prompt.

### Reward Signal

```
edit_distance  = 1 - difflib.SequenceMatcher.ratio(agent_text, doctor_text)
reward         = 1 - edit_distance        (0 = identical, 1 = completely different)
match_rate     = fraction of sections doctor left unchanged (< 5% change threshold)
```

### Mock Doctor Hidden Policy

The simulated reviewer applies these rules — unknown to the agent, learned only through observing corrections:

1. Standardise medications to generic name + dose + frequency
2. Add clinical precision to diagnoses (ICD-style language)
3. Normalise all dates to YYYY-MM-DD
4. Replace vague discharge conditions with specific clinical observations
5. Prefix all pending items with "PENDING:"
6. Append "(Requires clinician verification)" to MISSING fields
7. Ensure hospital course covers: presenting symptoms, interventions, response, and reason for discharge

### How Learning Works

```
Iteration 1:
  Agent extracts  → "heart failure"
  Doctor corrects → "Acute decompensated heart failure with reduced ejection fraction (HFrEF)"
  Stored in memory.

Iteration 2 prompt includes:
  [PRINCIPAL_DIAGNOSIS]
  Agent wrote : heart failure
  Doctor wrote: Acute decompensated heart failure with reduced ejection fraction (HFrEF)

  Agent now produces more specific language.
  Edit distance drops. Reward improves.
```

### Expected Improvement Curve

```
Iter   Reward   Edit Dist   Match Rate
   1    0.52      0.48        40%
   2    0.65      0.35        55%
   3    0.74      0.26        65%
   4    0.81      0.19        75%
   5    0.86      0.14        82%
```

---

## Design Decisions and Trade-offs

### BM25 vs Vector Store

| | BM25 (fallback) | ChromaDB + FastEmbed (primary) |
|---|---|---|
| Match type | Exact keyword | Semantic meaning |
| Setup | Zero dependencies | ~33MB model download |
| Quality | Misses synonyms | Finds "HFrEF" when searching "heart failure" |
| Memory | ~1MB | ~200MB (model loaded) |

Decision: ChromaDB + FastEmbed as primary, BM25 as automatic fallback.

### 1 Batch LLM Call vs 13 Sequential Calls

| | 13 sequential | 1 batch call |
|---|---|---|
| Latency | ~40–60s | ~8–12s |
| API cost | 13x | 1x |
| Risk | Reliable per-section | JSON parse can fail |

Decision: batch call with automatic per-section fallback if JSON parsing fails.

### Correction Memory vs Fine-tuning

| | Fine-tuning (DPO/SFT) | Correction Memory |
|---|---|---|
| Data needed | 100s–1000s pairs | Works from iteration 1 |
| Infrastructure | GPU, training loop | JSON file + prompt injection |
| Latency | Hours to train | Instantaneous |
| Auditability | Hard to inspect | Fully readable JSON |

Decision: correction memory injection — practical, auditable, zero infrastructure.

### OCR Batching Strategy

Processing scanned PDFs 3 pages at a time keeps peak RAM at ~3 page images instead of all 71 simultaneously. Cache is written after each batch so partial progress survives crashes.

---

## Limitations

### Part 1

- **OCR quality** — extraction accuracy depends on scan quality; poor scans produce more MISSING fields
- **Context window** — batching all 13 sections in one prompt increases truncation risk for very long notes
- **Mock tools** — DrugInteractionTool covers ~10 known pairs; production would call a clinical drug API
- **Single-patient isolation** — RAG collections are per-patient; cross-patient knowledge is not shared

### Part 2

- **Cold-start** — no correction memory exists on iteration 1; improvement begins only from iteration 2
- **Gaming risk** — optimising for edit distance can be exploited: vaguer text gets fewer edits but is not better medicine. Mitigated by monitoring section_match_rate separately from reward
- **Mock doctor fidelity** — the simulated reviewer applies a fixed LLM policy; real clinicians have patient-specific judgment this cannot capture
- **Safety guarantees** — correction memory only adds examples to the prompt; the hard MISSING/CONFLICT guardrails in `_apply_value` and `compile_summary` cannot be overridden by injected memory

### What Would Be Done With More Time

- Replace mock DrugInteractionTool with a real clinical drug interaction API
- Add cross-patient learning (shared correction memory across patients)
- Evaluate on a held-out patient set to measure generalisation
- Add section-level confidence scores to the output
- Implement streaming output so clinicians see sections as they are extracted
- Add a feedback UI for clinicians to accept/reject sections directly
