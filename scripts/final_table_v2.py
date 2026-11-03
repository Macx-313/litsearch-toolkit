#!/usr/bin/env python3
# N-9 终表 v2 合成：全臂统一 n=21（剔除错配 23000183）+ Wilson CI + pool 标注 + 脚注
import json, math, statistics

LT = f"{LT}/logs"
VOID = "23000183"

def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0, c-h), min(1, c+h))

def rd_of(spec):
    if "#" in spec:
        f, key = spec.split("#"); return json.load(open(f))[key]["rank_details"]
    return json.load(open(spec))["rank_details"]

ARMS = [
    ("dense_zh (Qwen3-8B, 中文直查)", "dense_top100",        f"{LT}/p1_gold_recall.json"),
    ("hybrid_en (Qwen3-8B+BM25 RRF)", "hybrid_RRF_top100",   f"{LT}/n9_hybrid_recall.json"),
    ("bge_hybridpool_en (生产精排)",   "hybrid_RRF→bge",      f"{LT}/rerank_v2_hybrid_pool.json#bge_hybridpool_en"),
    ("bge_densepool_en (对照)",        "dense→bge",           f"{LT}/n9_rerank.json#bge_en"),
    ("qwen3r_aligned (对照, 已拒)",    "hybrid_RRF→qwen3r",   f"{LT}/rerank_v2_hybrid_pool.json#qwen3r_aligned_hybridpool_en"),
    ("medcpt_768d (领域专训对照)",     "dense(MedCPT)",       f"{LT}/n9_medcpt_recall.json"),
]

table = {}
for name, pool, spec in ARMS:
    rd = {p: r for p, r in rd_of(spec).items() if p != VOID}
    n = len(rd)
    row = {"pool": pool, "n": n}
    for k in (1, 5, 10, 20, 100):
        hit = sum(1 for r in rd.values() if r <= k)
        lo, hi = wilson(hit, n)
        row[f"@{k}"] = f"{hit}/{n}={hit/n:.3f} [{lo:.2f},{hi:.2f}]"
    finite = [r for r in rd.values() if r < 999]
    row["median(finite)"] = statistics.median(finite) if finite else None
    row["miss"] = sum(1 for r in rd.values() if r == 999)
    table[name] = row
    print(name, "→", json.dumps(row, ensure_ascii=False))

out = {
    "_title": "N-9 终表 v2（全臂 n=21，剔除 qrels 错配 23000183，Wilson 95% CI）",
    "_footnotes": [
        "① 23000183 qrels 错配（WB C-1）：查询描述'快速消退'，目标文献实为'脊髓腔再分布+PPS'；已作废，多目标 qrels 列为扩样路线",
        "② 跨 pool 的行不可直接比较（WB M-3）：dense_*/bge_densepool 来自 dense top-100 池，hybrid_* 来自 hybrid RRF 池",
        "③ 29660550 查询后半句'与血肿再分布'属兄弟文献机制，本篇为水瘤转化——轻度措辞偏移，未改 n（WB M-4）",
        "④ C-3：n=21 下多数臂间差异未达统计显著（McNemar p≥0.06）；本表为'已知项回归测试'，非效能估计；扩样路线：38 种子全用+查询变体→n=60-90",
        "⑤ 语料口径：MeSH 索引子集内、已知在库目标（N-1/N-2：24.7% 题名-only 向量；快照差 2.4% 且扩大中）",
        "⑥ 精排器定型依据：预注册规则 v2（McNemar），bge vs qwen3r 配对 b=14:c=0 p<0.001；qwen3r 绝对值异常低不排除残余实现问题，已记录",
    ],
    "_preregistration": json.load(open(f"{LT}/rerank_v2_hybrid_pool.json"))["preregistration_log"],
    "arms": table,
}
json.dump(out, open(f"{LT}/n9_final_table_v2.json", "w"), ensure_ascii=False, indent=1)
print("saved -> n9_final_table_v2.json")
