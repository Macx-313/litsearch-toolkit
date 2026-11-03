import os
#!/usr/bin/env python3
# 重建 pubmed_v2（dense+sparse 双 named vector）→ 混合 RRF 召回对比
# 用法: python3 rebuild_v2_hybrid.py <dense_prefix> <subset.jsonl> [--skip-build]
import json, sys, time
import numpy as np
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models

LT = os.environ.get("LITSEARCH_HOME", os.path.expanduser("~/litsearch"))
QD = QdrantClient(url="http://localhost:6333", timeout=300)
COL = "pubmed_v2"

def main():
    vprefix, subset = sys.argv[1], sys.argv[2]
    skip_build = "--skip-build" in sys.argv
    vecs = np.load(vprefix + ".npy", mmap_mode="r")
    vpmids = [l.strip() for l in open(vprefix + ".pmids.txt")]
    texts = {}
    meta = {}
    for line in open(subset):
        d = json.loads(line)
        p = str(d["pmid"])
        t = (d.get("title") or "").strip(); a = (d.get("abstract") or "").strip()
        txt = (t + "\n" + a).strip() if a else t
        texts[p] = txt
        meta[p] = (d.get("year"), bool(d.get("abstract")))
    dim = vecs.shape[1]
    vecs = np.asarray(vecs)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    print(f"n={len(vpmids)} dim={dim}", flush=True)

    sm = SparseTextEmbedding(model_name="Qdrant/bm25")
    import pickle, os
    cache = f"{LT}/encoded/p1_sparse.pkl"
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            sparses = pickle.load(f)
        print(f"sparse cache loaded {len(sparses)}", flush=True)
    else:
        t0 = time.time()
        sparses = []
        for i, p in enumerate(vpmids):
            sv = next(sm.embed([texts[p]]))
            sparses.append(models.SparseVector(indices=[int(x) for x in sv.indices],
                                               values=[float(x) for x in sv.values]))
            if i % 50000 == 0:
                print(f"  sparse {i}/{len(vpmids)} ({time.time()-t0:.0f}s)", flush=True)
        print(f"sparse done {time.time()-t0:.0f}s", flush=True)
        with open(cache, "wb") as f:
            pickle.dump(sparses, f, protocol=4)

    if not skip_build:
        try:
            QD.delete_collection(COL)
        except Exception:
            pass
        QD.create_collection(
            COL,
            vectors_config={
                "dense": models.VectorParams(size=dim, distance=models.Distance.DOT,
                    quantization_config=models.ProductQuantization(
                        product=models.ProductQuantizationConfig(
                            compression=models.CompressionRatio.X4, always_ram=True))),
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)),
            },
            hnsw_config=models.HnswConfigDiff(m=16, ef_construct=128),
        )
        B = 2048
        for i in range(0, len(vpmids), B):
            j = min(i + B, len(vpmids))
            QD.upload_points(COL, points=[
                models.PointStruct(id=k,
                    vector={"dense": vecs[k].tolist(), "sparse": sparses[k]},
                    payload={"p": p, "y": meta[p][0], "a": meta[p][1]})
                for k, p in enumerate(vpmids[i:j], start=i)], wait=True)
            if i % (B * 25) == 0:
                print(f"  upload {j}/{len(vpmids)}", flush=True)
        print(f"uploaded points={QD.get_collection(COL).points_count}", flush=True)

    # 混合召回（英文查询：dense(en) + sparse(en) + RRF）
    qv = np.load(f"{LT}/goldset/gold_queries_en_enc.npy")
    qpmids = [l.strip() for l in open(f"{LT}/goldset/gold_queries_en_enc.pmids.txt")]
    qtexts = {r[0]: r[1] for r in (l.rstrip("\n").split("\t") for l in open(f"{LT}/goldset/gold_queries_en.tsv"))}
    pmid2id = {p: i for i, p in enumerate(vpmids)}
    ranks = []
    for qi, target in enumerate(qpmids):
        dq = (qv[qi] / np.linalg.norm(qv[qi])).tolist()
        sq = next(sm.embed([qtexts[target]]))
        sqv = models.SparseVector(indices=[int(x) for x in sq.indices],
                                  values=[float(x) for x in sq.values])
        r = QD.query_points(COL,
            prefetch=[models.Prefetch(query=dq, using="dense", limit=100),
                      models.Prefetch(query=sqv, using="sparse", limit=100)],
            query=models.FusionQuery(fusion=models.Fusion.RRF), limit=100).points
        ids = [p.id for p in r]
        tgt = pmid2id.get(target)
        ranks.append(ids.index(tgt) + 1 if tgt in ids else 999)
    ranks = np.array(ranks)
    res = {"arm": "hybrid_RRF_en", "n": len(qpmids),
           "recall@5": float((ranks <= 5).mean()), "recall@10": float((ranks <= 10).mean()),
           "recall@20": float((ranks <= 20).mean()), "recall@100": float((ranks <= 100).mean()),
           "median_rank": float(np.median(ranks)) if (ranks < 999).any() else None,
           "misses": int((ranks == 999).sum()),
           "rank_details": {p: int(r) for p, r in zip(qpmids, ranks)}}
    json.dump(res, open(f"{LT}/logs/n9_hybrid_recall.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "rank_details"}, ensure_ascii=False))

if __name__ == "__main__":
    main()
