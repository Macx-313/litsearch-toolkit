#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P0 窗口评估套件（host python 运行，读容器产出的向量文件）：
1) cosine 比对: p0_corpus_100k(bf16 主档) vs p0_10k_fp8 重叠 1 万条的 cosine 分布
2) MedCPT 对照: 需先产 medcpt_10k.npy（由 master 脚本调用本文件 --medcpt 模式生成）
用法:
  python3 p0_eval.py cosine <bf16.npz> <fp8_prefix>
  python3 p0_eval.py medcpt <corpus.jsonl> <out_prefix>   # CPU 编码 1 万条
依赖: numpy; medcpt 模式另需 sentence_transformers（embedding-env）
"""
import sys, os, json
import numpy as np

def load_prefix(p):
    vecs = np.load(p + ".npy")
    pmids = [l.strip() for l in open(p + ".pmids.txt")]
    return vecs, pmids

def cmd_cosine(bf16_prefix, fp8_prefix):
    vb, pb = load_prefix(bf16_prefix)
    vf, pf = load_prefix(fp8_prefix)
    idx_b = {p: i for i, p in enumerate(pb)}
    pair = [(idx_b[p], i) for i, p in enumerate(pf) if p in idx_b]
    print(f"overlap={len(pair)}")
    cs = []
    for ib, iff in pair:
        a, b = vb[ib].astype(np.float32), vf[iff].astype(np.float32)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 0 and nb > 0:
            cs.append(float(np.dot(a, b) / (na * nb)))
    cs = np.array(cs)
    print(f"cosine: mean={cs.mean():.6f} p5={np.percentile(cs,5):.6f} min={cs.min():.6f} frac>=0.99={float((cs>=0.99).mean()):.4f}")
    json.dump({"overlap": len(pair), "mean": float(cs.mean()), "p5": float(np.percentile(cs,5)),
               "min": float(cs.min()), "frac_ge_099": float((cs>=0.99).mean())},
              open(f"{LT}/logs/p0_cosine.json", "w"), indent=1)
    print("saved -> logs/p0_cosine.json")

def cmd_medcpt(infile, out_prefix, limit=10000):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("ncbi/MedCPT-Article-Encoder", device="cpu")
    pmids, texts = [], []
    with open(infile) as f:
        for line in f:
            d = json.loads(line)
            t = (d.get("title") or "").strip(); a = (d.get("abstract") or "").strip()
            txt = (t + "\n" + a).strip() if a else t
            if not txt: continue
            pmids.append(str(d["pmid"])); texts.append(txt)
            if len(pmids) >= limit: break
    print(f"medcpt encoding {len(pmids)} docs (CPU)...")
    v = m.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    np.save(out_prefix + ".npy", v.astype(np.float32))
    with open(out_prefix + ".pmids.txt", "w") as f:
        f.write("\n".join(pmids) + "\n")
    print("saved", out_prefix)

if __name__ == "__main__":
    if sys.argv[1] == "cosine":
        cmd_cosine(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "medcpt":
        cmd_medcpt(sys.argv[2], sys.argv[3])
