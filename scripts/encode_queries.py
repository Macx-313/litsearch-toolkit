#!/usr/bin/env python3
# 查询编码器（容器内 GPU，bf16，与文档侧同模型同精度——N-11 断言双侧打印）
# Qwen3-Embedding 查询侧必须带 instruction 前缀（文档侧免）——格式错误=静默降质，此处硬编码正确格式
import json, sys, time
import numpy as np
from vllm import LLM

INFILE, OUT = sys.argv[1], sys.argv[2]
MODEL = "Qwen/Qwen3-Embedding-8B"
INSTR = "Instruct: Given a biomedical research question, retrieve relevant English literature abstracts\nQuery: "

llm = LLM(model=MODEL, runner="pooling", max_model_len=512,
          gpu_memory_utilization=0.92, tensor_parallel_size=1)
print(f"[query-encode] MODEL={MODEL} dim={llm.get_embedding_dimension() if hasattr(llm,'get_embedding_dimension') else 'TBD'}", flush=True)

rows = [l.rstrip("\n").split("\t") for l in open(INFILE) if l.strip()]
pmids = [r[0] for r in rows]
texts = [INSTR + r[1] for r in rows]
out = llm.embed(texts, use_tqdm=False)
vecs = np.array([o.outputs.embedding for o in out], dtype=np.float32)
np.save(OUT + ".npy", vecs)
open(OUT + ".pmids.txt", "w").write("\n".join(pmids) + "\n")
print(f"[query-encode] DONE n={len(pmids)} dim={vecs.shape[1]}", flush=True)
