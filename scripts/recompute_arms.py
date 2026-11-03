#!/usr/bin/env python3
# C-1/X-2 落地：22 条查询自审结论 + 剔除错配条目后按 rank_details 重算各臂（n=21 + Wilson 95% CI）
import json, math

LT = f"{LT}/logs"
VOID = "23000183"  # WB 回执 C-1：查询（"快速自行消退"）与目标文献（向脊髓腔再分布+PPS）语义错配，qrels 作废

ARMS = {
    "dense_zh":  f"{LT}/p1_gold_recall.json",
    "hybrid_en": f"{LT}/n9_hybrid_recall.json",
    "bge_zh":    f"{LT}/n9_rerank.json#bge_zh",
    "bge_en":    f"{LT}/n9_rerank.json#bge_en",
}

def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0, c-h), min(1, c+h))

import statistics
out = {}
for arm, spec in ARMS.items():
    if "#" in spec:
        f, key = spec.split("#")
        rd = json.load(open(f))[key]["rank_details"]
    else:
        rd = json.load(open(spec))["rank_details"]
    rd21 = {p: r for p, r in rd.items() if p != VOID}
    n = len(rd21)
    ranks = sorted(rd21.values())
    row = {"n": n}
    for k in (1, 5, 10, 20, 100):
        hit = sum(1 for r in rd21.values() if r <= k)
        lo, hi = wilson(hit, n)
        row[f"recall@{k}"] = f"{hit}/{n}={hit/n:.3f} CI[{lo:.2f},{hi:.2f}]"
    finite = [r for r in rd21.values() if r < 999]
    row["median_rank(finite)"] = statistics.median(finite) if finite else None
    row["misses"] = sum(1 for r in rd21.values() if r == 999)
    out[arm] = row
    print(arm, json.dumps(row, ensure_ascii=False))

json.dump({"void_entry": VOID, "void_reason": "qrels 标注错配（WB C-1）：查询描述主流消退现象，目标文献实为脊髓腔再分布+PPS；dense top-10 对该查询 10/10 语义正确", "arms_n21": out},
          open(f"{LT}/n9_arms_v2_n21.json", "w"), ensure_ascii=False, indent=1)
print(f"saved -> {LT}/n9_arms_v2_n21.json")
