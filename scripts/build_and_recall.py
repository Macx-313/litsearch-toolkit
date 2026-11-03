import os
#!/usr/bin/env python3
# 建库（pq4 pinned——P0 矩阵胜者）+ gold set 召回测试
# 用法: python3 build_and_recall.py <vecs_prefix> <subset.jsonl> <gold_queries_prefix>
import json, sys, time
import numpy as np
from qdrant_client import QdrantClient, models

LT = os.environ.get("LITSEARCH_HOME", os.path.expanduser("~/litsearch"))
QD = QdrantClient(url="http://localhost:6333", timeout=300)
COL = "pubmed_v1"

def main():
    vprefix, subset, qprefix = sys.argv[1], sys.argv[2], sys.argv[3]
    vecs = np.load(vprefix + ".npy", mmap_mode="r")
    vpmids = [l.strip() for l in open(vprefix + ".pmids.txt")]
    meta = {}
    for line in open(subset):
        d = json.loads(line)
        meta[str(d["pmid"])] = (d.get("year"), bool(d.get("abstract")))
    missing = [p for p in vpmids if p not in meta]
    assert not missing, f"向量缺少元数据: {len(missing)} 条（如 {missing[:3]}）"
    # 反向多余（空文本被编码器跳过）无害：以向量侧为准
    if len(meta) != len(vpmids):
        print(f"note: 元数据 {len(meta)} 条 > 向量 {len(vpmids)} 条（空文本跳过 {len(meta)-len(vpmids)} 条，正常）")
    dim = vecs.shape[1]
    vecs = np.asarray(vecs)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    print(f"build {COL}: n={len(vpmids)} dim={dim}", flush=True)

    try:
        QD.delete_collection(COL)
    except Exception:
        pass
    QD.create_collection(
        COL,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.DOT),
        quantization_config=models.ProductQuantization(
            product=models.ProductQuantizationConfig(compression=models.CompressionRatio.X4, always_ram=True)),
        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=128),
    )
    B = 2048
    for i in range(0, len(vpmids), B):
        j = min(i + B, len(vpmids))
        QD.upload_points(COL, points=[
            models.PointStruct(id=k, vector=vecs[k].tolist(),
                               payload={"p": p, "y": meta[p][0], "a": meta[p][1]})
            for k, p in enumerate(vpmids[i:j], start=i)], wait=True)
        if i % (B * 25) == 0:
            print(f"  upload {j}/{len(vpmids)}", flush=True)
    info = QD.get_collection(COL)
    print(f"uploaded points={info.points_count}", flush=True)

    # gold 召回
    qv = np.load(qprefix + ".npy")
    qpmids = [l.strip() for l in open(qprefix + ".pmids.txt")]
    pmid2id = {p: i for i, p in enumerate(vpmids)}
    ranks = []
    for qi, target in enumerate(qpmids):
        r = QD.query_points(COL, query=(qv[qi] / np.linalg.norm(qv[qi])).tolist(), limit=100).points
        ids = [p.id for p in r]
        tgt = pmid2id.get(target)
        rank = ids.index(tgt) + 1 if tgt in ids else 999
        ranks.append(rank)
    ranks = np.array(ranks)
    res = {"n_queries": len(qpmids), "recall@5": float((ranks <= 5).mean()),
           "recall@10": float((ranks <= 10).mean()), "recall@20": float((ranks <= 20).mean()),
           "recall@50": float((ranks <= 50).mean()), "recall@100": float((ranks <= 100).mean()),
           "median_rank": float(np.median(ranks)) if (ranks < 999).any() else None,
           "misses": int((ranks == 999).sum()),
           "rank_details": {p: int(r) for p, r in zip(qpmids, ranks)}}
    json.dump(res, open(f"{LT}/logs/p1_gold_recall.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "rank_details"}, ensure_ascii=False))

if __name__ == "__main__":
    main()
