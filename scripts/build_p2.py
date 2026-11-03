import os
#!/usr/bin/env python3
# P2 全量建库：免合并直传双半（memmap 流式，RAM 占用 = 批次级），pubmed_full（dense pq4 pinned）
# 用法: python3 build_p2.py [--from-half 0|1]  （断点：--from-half 1 表示 half0 已传完）
import json, sys, time
import numpy as np
from qdrant_client import QdrantClient, models

LT = os.environ.get("LITSEARCH_HOME", os.path.expanduser("~/litsearch"))
TOTAL = 28337072
QD = QdrantClient(url="http://localhost:6333", prefer_grpc=True, timeout=600)
COL = "pubmed_full"
B = 8192

def main():
    from_half = int(sys.argv[sys.argv.index("--from-half") + 1]) if "--from-half" in sys.argv else 0
    pmid_year = {}
    for line in open(f"{LT}/parsed/full_corpus.jsonl"):
        d = json.loads(line)
        pmid_year[str(d["pmid"])] = d.get("year")
    print(f"corpus meta loaded: {len(pmid_year)}", flush=True)

    try:
        QD.delete_collection(COL)
    except Exception:
        pass
    QD.create_collection(
        COL,
        vectors_config=models.VectorParams(size=4096, distance=models.Distance.DOT,
            quantization_config=models.ProductQuantization(
                product=models.ProductQuantizationConfig(
                    compression=models.CompressionRatio.X4, always_ram=True))),
        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=128),
    )

    gid = 0
    t0 = time.time()
    for half in (0, 1):
        if half < from_half:
            # 已传过：仅推进全局 id
            n0 = sum(1 for _ in open(f"{LT}/encoded/p2_half{half}.pmids.txt"))
            gid += n0
            continue
        vecs = np.load(f"{LT}/encoded/p2_half{half}.npy", mmap_mode="r")
        pmids = [l.strip() for l in open(f"{LT}/encoded/p2_half{half}.pmids.txt")]
        assert vecs.shape[0] == len(pmids), f"half{half} 向量/PMID 不一致 {vecs.shape[0]} vs {len(pmids)}"
        for i in range(0, len(pmids), B):
            j = min(i + B, len(pmids))
            batch = np.asarray(vecs[i:j], dtype=np.float32)
            batch = batch / np.linalg.norm(batch, axis=1, keepdims=True)
            QD.upload_points(COL, points=[
                models.PointStruct(id=gid + (k - i), vector=batch[k - i].tolist(),
                                   payload={"p": pmids[k], "y": pmid_year.get(pmids[k])})
                for k in range(i, j)], wait=True)
            gid += j - i
            if (i // B) % 100 == 0:
                el = time.time() - t0
                print(f"  half{half} {j}/{len(pmids)} total={gid} ({gid/el:.0f}/s eta={(28337072-gid)/max(gid/el,1)/3600:.1f}h)", flush=True)
    print(f"uploaded total_id={gid}", flush=True)
    # 上传后强制对账（本次事故教训：done 标记不等于数据完整性）
    time.sleep(120)  # 等 WAL 应用
    cnt = QD.count(collection_name=COL, exact=True).count
    print(f"[verify] exact count={cnt} (expect {TOTAL}, 差={TOTAL-cnt})", flush=True)
    ok = cnt == TOTAL
    # 抽查首/中/尾 id 与 payload 对齐
    for gid_chk in [0, TOTAL // 2, TOTAL - 1]:
        try:
            r = QD.retrieve(collection_name=COL, ids=[gid_chk], with_payload=True)
            print(f"[verify] id={gid_chk} -> {r[0].payload if r else 'MISSING'}", flush=True)
            ok = ok and bool(r)
        except Exception as e:
            print(f"[verify] id={gid_chk} error: {e}", flush=True); ok = False
    json.dump({"done": True, "verified": ok, "points": cnt, "total_id": gid, "ts": time.strftime("%F %T")},
              open(f"{LT}/logs/p2_build_done.json", "w"))
    print(f"[verify] {'ALL OK' if ok else '!!MISMATCH——勿启动终评，人工核查'}", flush=True)

if __name__ == "__main__":
    main()
