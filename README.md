# PubMed Semantic Search Toolkit

A complete pipeline for building a local semantic search engine over **28+ million PubMed abstracts** using the open-source Qwen3-Embedding-8B model. Query in natural language (Chinese or English), get back relevant PMIDs in milliseconds.

## Performance (measured, independently verified)

Evaluation set: 31 known target papers (aSDH/sICH seeds), each described as 3 query variants (Chinese formal / Chinese colloquial / English technical) = 93 queries. Not a formal academic benchmark — it is an engineering self-check built from the author's own systematic-review seeds. See `eval/` for the full data.

| Metric | Value |
|---|---|
| Corpus | 28,337,072 abstracts (PubMed 2026-01 baseline, records with abstract; 70.9% of 40.0M total) |
| Model | Qwen3-Embedding-8B (Apache-2.0, 4096-dim) |
| Index | Qdrant pq4 quantization + HNSW, ~58GB in RAM |
| Known-item recall@100 (in-corpus targets) | **28/28 = 100%** |
| Known-item recall@100 (all 31 targets incl. 3 out-of-corpus) | **28/31 ≈ 90.3%** |
| Keyword baseline, same 31 targets (PubMed relevance sort, top-100) | **16/31 = 51.6%** |
| Paired McNemar | b=12, c=0, **p = 0.0005** |
| Query latency (default) | ~281ms p50 |
| Query latency (fast mode) | ~32ms p50 |
| End-to-end single query | ~4s (including CPU encoding) |

The 3 missed targets have no abstract in PubMed (title-only records) and are therefore not in the corpus by design — the corpus only indexes abstracts.

## Architecture

```
PubMed Baseline XML (51GB, NLM official)
  ↓ parse_all.py (parallel, 16 workers)
full_corpus.jsonl (40GB, 28.3M records)
  ↓ encode_corpus.py (Qwen3-Embedding-8B, DP2 dual-GPU)
fp32 vectors (432GiB) + pmid lists
  ↓ build_p2.py (Qdrant gRPC upload + forced verification)
Qdrant collection (pq4 + HNSW, GREEN)
  ↑ query: litsearch.sh → :8004 encode → :6333 search → pmid_offset.db → abstracts
```

## Quick Start (reproduce from scratch)

### Prerequisites
- Linux server with 2× GPU (48GB+ VRAM each) for encoding, 150GB+ RAM for serving
- Docker (for vLLM encoding container)
- Python 3.12+ with: `qdrant-client`, `fastembed`, `sentence-transformers`, `numpy`, `transformers`
- Qdrant 1.19+ (Docker container, data on large NVMe)

### Step 1: Download PubMed baseline (~51GB)
```bash
mkdir -p /data/pubmed/baseline
# List of files at https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/
# Download pubmed26n0001.xml.gz through pubmed26n1334.xml.gz
```

### Step 2: Parse to JSONL (16-way parallel, ~20 min)
```bash
python3 scripts/parse_all_runner.sh
# Output: parsed/full_corpus.jsonl (~40GB, 28.3M records)
```

### Step 3: Encode with Qwen3-Embedding-8B (DP2 dual-GPU, ~4.5 days)
```bash
# Split corpus in half, encode each half on one GPU
bash scripts/litsearch_b_window.sh "bash scripts/master_p2.sh"
# Output: encoded/p2_half0.npy + p2_half1.npy (432GiB total)
```

### Step 4: Build Qdrant index (~9h with gRPC)
```bash
python3 scripts/build_p2.py
# Includes forced verification: exact count must equal corpus count
```

### Step 5: Deploy query services
```bash
# Query encoder (CPU, always-on): scripts/litsearch_query_server.py → :8004
# Search client: bash litsearch.sh "your query here" 20
```

## File Guide

| Category | Scripts | Purpose |
|---|---|---|
| **Data pipeline** | `parse_baseline.py` / `parse_all.py` / `parse_all_runner.sh` / `merge_subset.py` | XML → JSONL, MeSH filtering |
| **Encoding** | `encode_corpus.py` / `encode_queries.py` / `dl_models.py` | vLLM offline batch encoding, checkpointed |
| **Index build** | `build_p2.py` / `build_and_recall.py` / `rebuild_v2_hybrid.py` | Qdrant upload + verification + hybrid (dense+sparse) |
| **Query** | `litsearch_query_server.py` / `litsearch.py` | Standing encoder service + client |
| **Evaluation** | `final_eval_parallel.py` / `keyword_baseline.py` / `latency_anchor.py` | Known-item recall, latency benchmarks |
| **GPU orchestration** | `litsearch_b_window.sh` / `litsearch_b_patrol.sh` / `vllm_guard.sh` | Safe GPU windows with assertions + auto-restore |
| **Maintenance** | `build_offset_index.py` / `repair_p2.py` / `green_watch.sh` | PMID→offset index, point repair, optimization monitor |
| **MeSH expansion** | `expand_mesh.py` / `fetch_mesh_pmids.py` | MeSH subtree expansion for domain filtering |

## Key Design Decisions (documented in `docs/`)

- **Qwen3-Embedding-8B** selected as verified open-source SOTA (MTEB eng v2 75.22); Conan-v2 excluded (no weights), NV-Embed-v2 excluded (non-commercial license)
- **pq4 quantization** (4-bit product quantization): 100% recall retention at 1/8 memory; int8/pq8/pq16/pq32/binary all benchmarked
- **DP2 (data-parallel dual-GPU)** over TP2: 41 docs/s/card vs 20.8 docs/s TP2 — 2× speedup when model fits single card
- **Hybrid retrieval** (dense + BM25 sparse, RRF fusion) as default; bge-reranker-v2-m3 for re-ranking
- **Forced verification gate** after every bulk upload: exact count must equal expected, else abort (learned from a 4M-point loss incident)

## Pre-computed Vectors (optional, saves ~216 GPU·hours)

The full 28.3M fp32 embedding vectors (~432GiB) are available for download, eliminating the 4.5-day GPU encoding step:
- `p2_half0.npy` + `p2_half0.pmids.txt` (~216GiB)
- `p2_half1.npy` + `p2_half1.pmids.txt` (~216GiB)

See `DATA_SHARING.md` for legal basis, verification, and usage terms.
Distributed under Apache-2.0 (same as the embedding model), consistent with the SPECTER2 precedent (AllenAI, 120M paper embeddings).

**Option A — Pre-built Qdrant index (~569GB)**: download, point a Qdrant container at it, immediately searchable. Zero compute.
**Option B — Raw fp32 vectors (~432GiB)**: run `build_p2.py` (~9h CPU), choose your own quantization/index params.

Either way, you'll also need the text layer (abstracts) for displaying search results:
1. Download PubMed baseline from NLM (~51GB, freely available)
2. Run `parse_all.py` + `build_offset_index.py` (~23 min total)

## Documentation

- `docs/本地文献语义检索系统_整体规划_v1.3.md` — Full architecture & technology selection (with evidence)
- `docs/本地文献语义检索系统_全案收尾报告_v1.md` — Final delivery report (all metrics + incident archive)
- `eval/gold_queries_v93.tsv` — 93 evaluation queries (31 targets × 3 forms: Chinese / colloquial / English)

## License

- **Code**: Apache-2.0
- **Documentation**: CC BY 4.0
- **Data prerequisite**: PubMed baseline is freely available from NLM (attribution required, no NLM endorsement); Qwen3-Embedding-8B is Apache-2.0

## Known Limitations

- Title-only records (~11.7M) excluded from encoding (no abstract → weaker semantic signal)
- Snapshot frozen at 2026-01 baseline; incremental updates not yet automated
- Query encoding on CPU takes 1-4s per query (GPU encoder requires stopping the production vLLM service)
- Chinese colloquial queries have ~29% @1 recall (semantic ceiling of the model); agent-layer query rewriting recommended

---

Built over 13 days (2026-09-25 to 2026-10-07) with iterative independent review. All performance numbers are measured and reproducible from included scripts.
