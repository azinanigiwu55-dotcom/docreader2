# private-doc-qa

**Private document Q&A for professional-services firms.** Ask questions about client documents and get answers that cite the exact source, respect who is allowed to read what, and never leave your environment.

> Status: early prototype (v0.1). Runs entirely on synthetic data. Not production software.

## Why this exists

Accounting, legal and consulting firms hold confidential client files and want the benefit of AI search, but pasting those files into a public chatbot is not an option. Most "chat with your documents" demos skip the three things a compliance-minded firm asks about first:

1. **Permission-aware retrieval.** If a user cannot open a document, the AI must not use it to answer them.
2. **Citations and honest refusals.** Every answer points to a document and page, and the system says "I couldn't find this" instead of guessing.
3. **An audit trail.** Who asked what, and which documents were used.

This project builds those three things first and measures them.

## What works today

| Feature | Status |
|---|---|
| PDF, DOCX, TXT/MD ingestion with page/section metadata | Done |
| Hybrid search (BM25 keyword + embeddings, fused with reciprocal rank fusion) | Done |
| Per-document access groups, enforced **before** ranking | Done |
| Citation-required answers; uncited answers are rejected | Done |
| "Not found" abstention | Done |
| Prompt-injection test document in the evaluation set | Done |
| JSONL audit log | Done |
| Evaluation harness (accuracy, citations, refusals, access leaks, latency) | Done |
| Pluggable models: Hugging Face (Colab GPU), Ollama (laptop/VPS), offline baseline | Done |
| pgvector store, SharePoint connector, Entra ID login, Terraform deployment | Roadmap |

## Quick start

### Option A: Google Colab (free GPU, no credit card)

1. Push this repo to your GitHub account.
2. Open `notebooks/01_colab_pipeline.ipynb` in Colab and set a **T4 GPU** runtime.
3. Edit `REPO_URL` in the first cell and run the cells in order.

### Option B: your own machine or a plain Linux VPS (no Docker)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[embeddings,dev]"

# 1. generate synthetic documents, access rules and test questions
docqa synth --out data

# 2. ingest, chunk, embed and index
docqa index --docs data/docs --acl data/acl.json --out data/index

# 3. ask (needs a model: see "Choosing a model" below)
docqa ask --index data/index --backend ollama --groups partners \
  "What is the annual audit fee for Karoo Fresh Produce (Pty) Ltd?"

# 4. evaluate
docqa eval --index data/index --backend ollama --questions data/questions.json
```

Run the tests with `pytest`. They need no model download.

### Choosing a model

| Where you run | Backend | Suggested model |
|---|---|---|
| Colab (T4 GPU) | `hf` | `Qwen/Qwen2.5-3B-Instruct`, or the 7B with `--four-bit` |
| CPU laptop or VPS (16 GB RAM) | `ollama` | a 3-4B model, e.g. `qwen2.5:3b` |
| Quick smoke test, no model | `extractive` | built-in baseline (not a real LLM) |

Install Ollama separately and run `ollama pull qwen2.5:3b` before using the `ollama` backend. Use `--embedder hash` to skip the embedding download in offline tests.

## How it works

```
documents --> parse + chunk --> index (BM25 + embeddings)
                 |                       |
        page/section + ACL        search(query, user_groups)
                                         |
                       only chunks the user may read, top-k
                                         |
                    LLM answers with [n] citations, or NOT_FOUND
                                         |
              no valid citation? -> "couldn't find this"
                                         |
                              answer + sources + audit log
```

### Design decisions

- **Filter before ranking.** Restricted chunks are removed before search, so they cannot reach the model or influence scores. The evaluation includes a leak check.
- **Hybrid search.** Keyword matching catches exact figures, clause numbers and names; embeddings catch paraphrases. Contracts and financial documents need both.
- **Title-aware indexing.** The document title is part of the indexed text, which helps separate near-identical documents such as one engagement letter per client.
- **Sources are untrusted.** The system prompt tells the model to ignore instructions inside documents, and the evaluation set includes a poisoned document to test that.
- **Small, swappable store.** The in-memory index exposes `build / search / save / load`. Replacing it with Postgres + pgvector means reimplementing one class.

## Evaluation

The synthetic corpus is generated from templates, so every question has a known answer. Question kinds:

| Kind | Passes when |
|---|---|
| `answerable` | the answer contains the expected fact |
| `injection` | same, despite a document telling the model to lie |
| `unanswerable` | the system says it can't find the answer |
| `forbidden` | a user without access gets a refusal **and** no restricted chunk is retrieved |

Secondary metrics: retrieval hit rate (did the right document reach the model), citation correctness, latency.

### Results

Offline smoke-test baseline (no LLM, hash embeddings, seed 7). It only proves the pipeline and checks retrieval and access control; it says nothing about model quality.

| Setup | Answerable | Injection | Unanswerable | Forbidden | Retrieval hit | Access leaks |
|---|---|---|---|---|---|---|
| Extractive baseline | 25/34 | 1/1 | 5/5 | 2/3 | 100% | 0 |

Fill in real model results after your Colab run:

| Setup | Answerable | Injection | Unanswerable | Forbidden | Citation correct | Latency (mean) |
|---|---|---|---|---|---|---|
| Qwen2.5-3B, bge-small | | | | | | |
| Qwen2.5-7B (4-bit), bge-small | | | | | | |

Change one thing at a time (chunk size, `k`, model, prompt), re-run, and record what moved.

## Repository layout

```
src/docqa/
  synth.py     synthetic documents, access rules, evaluation questions
  ingest.py    parsing and chunking with page/section and ACL metadata
  index.py     hybrid index, embedders, permission filtering
  answer.py    prompts, model backends, citation checks, audit log
  eval.py      evaluation harness and report
  cli.py       command line (synth, index, ask, eval)
notebooks/     Colab notebook
tests/         pytest suite (no downloads needed)
```

## Roadmap

1. Measure and improve quality with real models (chunking, reranking, prompt).
2. Postgres + pgvector store behind the same interface.
3. Microsoft 365 / SharePoint connector that reads real folder permissions.
4. Microsoft Entra ID sign-in and a simple chat UI with source previews.
5. Audit dashboard on top of the JSONL log.
6. Terraform module to deploy into a customer's own Azure tenant.
7. Stronger prompt-injection defences and rate limiting.

## Security and privacy notes

- All data in this repo is **synthetic and fictional**. Do not commit real client or employer documents, credentials or API keys. `data/` and `.env` are git-ignored.
- Files missing from `acl.json` default to readable by everyone. Change `default_acl` in `ingest_folder` to `["admin"]` for default-deny.
- If you run this on a VPS, use SSH keys, a firewall, and keep Postgres and Ollama bound to localhost.
- Keeping data inside your own environment is an architectural property of this design. It is **not** a legal compliance claim. Anything about POPIA, GDPR or professional-confidentiality obligations needs review by a qualified professional.
- This is a personal project. If you work for an employer in this industry, check your contract's IP and outside-work clauses before building or publishing.

## License

Not yet chosen. Add a LICENSE file (MIT or Apache-2.0 are common choices) before sharing the repo publicly.
