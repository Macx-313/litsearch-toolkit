#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qdrant 量化矩阵负载测试（WB e2②：小集筛 8 组）
对 P0 向量子集建临时 collection，测：建库/灌库耗时、p50 延迟、top-10 召回保持率（vs 暴力精确检索）
用法: python3 qdrant_loadtest.py <vecs_prefix> [--n 10000]
"""
import sys, json, time
import numpy as np
from qdrant_client import QdrantClient, models

QD = QdrantClient(url="http://localhost:6333", timeout=120)
OUT = f"{LT}/logs/qdrant_loadtest.json"

def vec_cfg(kind):
    if kind.startswith("int8"):
        q = models.ScalarQuantization(scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8, quantile=0.99, always_ram=kind.endswith("pin")))
    elif kind.startswith("pq"):
        comp = {"pq4": models.CompressionRatio.X4, "pq8": models.CompressionRatio.X8,
                "pq16": models.CompressionRatio.X16, "pq32": models.CompressionRatio.X32}[kind.split("_")[0]]
        q = models.ProductQuantization(product=models.ProductQuantizationConfig(
            compression=comp, always_ram=kind.endswith("pin")))
    elif kind.startswith("bin"):
        q = models.BinaryQuantization(binary=models.BinaryQuantizationConfig(
            always_ram=kind.endswith("pin")))
    return dict(quantization_config=q)

CONFIGS = ["int8_pin", "int8_cold", "pq4_pin", "pq4_cold", "pq8_cold", "pq16_pin", "pq32_pin", "bin_pin"]

def main():
    prefix = sys.argv[1]
    n = int(sys.argv[sys.argv.index("--n")+1]) if "--n" in sys.argv else 10000
    vecs = np.load(prefix + ".npy")[:n].astype(np.float32)
    pmids = [l.strip() for l in open(prefix + ".pmids.txt")][:n]
    dim = vecs.shape[1]
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    QN = 200
    qs = vecs[:QN]
    base_top = np.argpartition(-(qs @ vecs.T), 10, axis=1)[:, :10]

    results = []
    for kind in CONFIGS:
        cname = "lt_" + kind.replace("_", "")
        try:
            QD.delete_collection(cname)
        except Exception:
            pass
        row = {"config": kind}
        try:
            t0 = time.time()
            QD.create_collection(cname, vectors_config=models.VectorParams(size=dim, distance=models.Distance.DOT),
                                 **vec_cfg(kind))
            row["create_s"] = round(time.time()-t0, 1)
            t1 = time.time()
            QD.upload_points(cname, points=[
                models.PointStruct(id=i, vector=vecs[i].tolist(), payload={"p": pmids[i]})
                for i in range(n)], wait=True)
            row["upload_s"] = round(time.time()-t1, 1)
            lat, hit = [], 0
            for qi in range(QN):
                ts = time.time()
                r = QD.query_points(cname, query=qs[qi].tolist(), limit=10,
                                    search_params=models.SearchParams(exact=False)).points
                lat.append(time.time()-ts)
                hit += len({p.id for p in r} & set(base_top[qi].tolist()))
            row["latency_ms_p50"] = round(float(np.median(lat))*1000, 1)
            row["recall@10_vs_exact"] = round(hit/(QN*10), 4)
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        try:
            QD.delete_collection(cname)
        except Exception:
            pass
    json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=1)
    print("saved ->", OUT)

if __name__ == "__main__":
    main()
