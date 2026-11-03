#!/usr/bin/env python3
import os, time
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import snapshot_download
MODELS = [
    "Qwen/Qwen3-Embedding-8B",
    "ncbi/MedCPT-Article-Encoder",
    "ncbi/MedCPT-Query-Encoder",
    "Qwen/Qwen3-Reranker-0.6B",
    "BAAI/bge-reranker-v2-m3",
]
for m in MODELS:
    print(f"[{time.strftime('%F %T')}] downloading {m}", flush=True)
    try:
        p = snapshot_download(m)
        print(f"OK -> {p}", flush=True)
    except Exception as e:
        print(f"FAIL {m}: {e}", flush=True)
print("ALL_DONE", flush=True)
