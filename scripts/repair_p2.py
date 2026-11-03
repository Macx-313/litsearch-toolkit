import os
#!/usr/bin/env python3
# P2 建库修复：scroll 现存 id → 差集 → 从半库 npy 按 id 补传缺失点 → 复核计数
# id 映射（与 build_p2 一致）：id = full_corpus 行号；id<H0 → half0[id]，否则 half1[id-H0]
import json, sys, time
import numpy as np
from qdrant_client import QdrantClient, models

LT = os.environ.get("LITSEARCH_HOME", os.path.expanduser("~/litsearch"))
QD = QdrantClient(url="http://localhost:6333", prefer_grpc=True, timeout=600)
COL = "pubmed_full"
H0 = 14168536
TOTAL = 28337072

def main():
    # 1) 现存 id 集合
    t0 = time.time()
    existing = set()
    offset = None
    while True:
        pts, offset = QD.scroll(collection_name=COL, limit=1024, offset=offset,
                                with_payload=False, with_vectors=False)
        existing.update(p.id for p in pts)
        if not pts or offset is None:
            break
    print(f"[repair] existing={len(existing)} ({time.time()-t0:.0f}s)", flush=True)
    missing = sorted(set(range(TOTAL)) - existing)
    print(f"[repair] missing={len(missing)}", flush=True)
    json.dump(missing[:50], open(f"{LT}/logs/repair_missing_head.json", "w"))
    if not missing:
        print("[repair] nothing to do")
        return

    # 2) pmid/year 元数据（full_corpus 单遍）
    pmid_year = {}
    with open(f"{LT}/parsed/full_corpus.jsonl") as f:
        for lineno, line in enumerate(f):
            p = line.split('"pmid": ', 1)[1].split(',', 1)[0]
            pmid_year[lineno] = (p, None)
    # year 从轻量重扫（只扫缺失 id 对应行不可能流式，改全扫 year 字段）
    del pmid_year
    pmids_years = [None] * TOTAL
    with open(f"{LT}/parsed/full_corpus.jsonl") as f:
        for lineno, line in enumerate(f):
            seg = line.split('"pmid": ', 1)[1].split(',', 1)[0]
            ypart = line.split('"year": ', 1)[1].split(',', 1)[0] if '"year": ' in line else "null"
            pmids_years[lineno] = (seg, None if ypart == "null" else int(ypart))
    print(f"[repair] meta loaded ({time.time()-t0:.0f}s)", flush=True)

    # 3) 半库向量 memmap + 补传
    v0 = np.load(f"{LT}/encoded/p2_half0.npy", mmap_mode="r")
    v1 = np.load(f"{LT}/encoded/p2_half1.npy", mmap_mode="r")
    assert v0.shape[0] == H0 and v1.shape[0] == TOTAL - H0
    B = 4096
    t1 = time.time()
    for i in range(0, len(missing), B):
        ids = missing[i:i + B]
        pts = []
        vecs = []
        for gid in ids:
            row = v0[gid] if gid < H0 else v1[gid - H0]
            vecs.append(np.asarray(row, dtype=np.float32))
        vecs = np.stack(vecs)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        for k, gid in enumerate(ids):
            p, y = pmids_years[gid]
            pts.append(models.PointStruct(id=gid, vector=vecs[k].tolist(), payload={"p": p, "y": y}))
        QD.upsert(collection_name=COL, points=pts, wait=True)
        if (i // B) % 50 == 0:
            done = i + len(ids)
            el = time.time() - t1
            print(f"  repaired {done}/{len(missing)} ({done/el:.0f}/s eta={(len(missing)-done)/max(done/el,1)/60:.0f}min)", flush=True)
    # 4) 复核
    time.sleep(30)
    cnt = QD.count(collection_name=COL, exact=True).count
    print(f"[repair] final exact count={cnt} (expect {TOTAL})", flush=True)
    json.dump({"repaired": len(missing), "final_count": cnt, "ts": time.strftime("%F %T")},
              open(f"{LT}/logs/p2_repair_done.json", "w"))

if __name__ == "__main__":
    main()
