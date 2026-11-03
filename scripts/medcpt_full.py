import os
#!/usr/bin/env python3
# MedCPT 对照臂（C-1 闭环件）：CPU 编码全子集 529k + 查询（英文）→ 建临时 collection → 召回
import json, sys, time
import numpy as np
import torch
torch.set_num_threads(48)
from sentence_transformers import SentenceTransformer

LT = os.environ.get("LITSEARCH_HOME", os.path.expanduser("~/litsearch"))

def main():
    t0 = time.time()
    art = SentenceTransformer("ncbi/MedCPT-Article-Encoder", device="cpu")
    qenc = SentenceTransformer("ncbi/MedCPT-Query-Encoder", device="cpu")
    print(f"[medcpt] models loaded {time.time()-t0:.0f}s", flush=True)

    # 查询（英文）
    rows = [l.rstrip("\n").split("\t") for l in open(f"{LT}/goldset/gold_queries_en.tsv")]
    qpmids = [r[0] for r in rows]
    qv = qenc.encode([r[1] for r in rows], batch_size=32, normalize_embeddings=True)
    np.save(f"{LT}/goldset/medcpt_qvecs.npy", qv.astype(np.float32))
    open(f"{LT}/goldset/medcpt_qvecs.pmids.txt", "w").write("\n".join(qpmids) + "\n")

    # 全子集编码（断点续跑：按行号 progress）
    texts, pmids = [], []
    for line in open(f"{LT}/parsed/p1_neurocrit.jsonl"):
        d = json.loads(line)
        t = (d.get("title") or "").strip(); a = (d.get("abstract") or "").strip()
        txt = (t + "\n" + a).strip() if a else t
        if not txt: continue
        pmids.append(str(d["pmid"])); texts.append(txt)
    n = len(texts)
    prog = f"{LT}/goldset/medcpt_full.progress"
    done = int(open(prog).read()) if __import__("os").path.exists(prog) else 0
    B = 512
    out = np.lib.format.open_memmap(f"{LT}/goldset/medcpt_full.npy", mode="r+" if done else "w+",
                                    dtype=np.float32, shape=(n, 768))
    t1 = time.time()
    for i in range(done, n, B):
        j = min(i + B, n)
        out[i:j] = art.encode(texts[i:j], batch_size=64, show_progress_bar=False,
                              normalize_embeddings=True)
        if (i // B) % 100 == 0:
            open(prog, "w").write(str(j))
            el = time.time() - t1
            print(f"[medcpt] {j}/{n} rate={(j-done)/el:.1f}docs/s eta={(n-j)/max((j-done)/el,0.1)/60:.0f}min", flush=True)
    open(prog, "w").write(str(n))
    out.flush()
    open(f"{LT}/goldset/medcpt_full.pmids.txt", "w").write("\n".join(pmids) + "\n")
    print(f"[medcpt] DONE {n} docs in {(time.time()-t1)/60:.0f}min", flush=True)

if __name__ == "__main__":
    main()
