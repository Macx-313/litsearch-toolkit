#!/usr/bin/env python3
# H-10/H-11/H-12 落实：终表表述修正 + 脚注⑦补预算/⑧正确配对 + 81.5% 子集稳健性检查（零成本）
import json, math

LT = f"{LT}/logs"
VOID = "23000183"

d = json.load(open(f"{LT}/n9_final_table_v2.json"))
rd = d["arms"]

# H-12 ①: 脚注⑦ 补输入预算差
fn7_old = [f for f in d["_footnotes"] if f.startswith("⑦")]
if fn7_old:
    d["_footnotes"].remove(fn7_old[0])
d["_footnotes"].append("⑦ H-4 闭合：dense_en 与 medcpt_768d 同为英文+纯 dense，唯一主变量=模型。残余变量披露：8B 侧 ≤1000 token vs MedCPT 侧 ≤512 token（max_position_embeddings 硬限），差额约束约 18.5% 的文档（>2000 字符者）；量化索引差异 pq4 vs int8 ≤1pp（矩阵实测）。稳健性检查见 _robustness_short_docs")

# H-11: 脚注⑧（WB 给定文本）
d["_footnotes"].append("⑧ 各因素的真实贡献须用正确配对臂读取：隔离稀疏腿 = dense_en vs hybrid_en（@5 17 vs 17，p=1.000，不可区分）；隔离翻译前置 = dense_zh vs dense_en（@10 15 vs 18，p=0.250，不显著）；bge 精排增益 = hybrid_en vs bge_hybridpool_en（p=1.000）。本表唯一达统计显著的对照是 dense_en vs medcpt_768d（@1 p=0.039、@100 p=0.031）。dense_zh 与 hybrid_* 的相邻行含查询语言差异，不得据其差值为稀疏腿/翻译归因")

# H-10: 定论改写（默认配置·不劣原则）
d["_conclusions"] = {
    "① 生产默认配置": "hybrid RRF → bge——依据是它在所有 @k 上都不劣于任何其他臂（@1/@5/@10 = 71.4/85.7/95.2%，全表最高或并列最高），而非'显著更优'（n=21 下各增益 p=1.000，未达区分度）；精排器选择已按预注册规则（McNemar）独立定型",
    "② 本阶段最硬结论": "通用 8B 显著优于领域专训 MedCPT（dense_en vs medcpt：@1 p=0.039、@100 p=0.031，n=21 单变量口径）——规划 C-1 闸门本地实测闭环",
    "③ 待扩样议题": "稀疏腿/翻译前置/bge 精排的真实增益在 n=21 下不可区分（正确配对 p=1.000/0.250/1.000），留待扩样（n=60-90，多目标 qrels）后判定；当前生产配置按'不劣'原则维持",
}

# H-12 ②: 目标文献 ≤2000 字符子集（~81.5%）上重算 dense_en vs medcpt
texts_len = {}
for line in open(f"{LT}/parsed/p1_neurocrit.jsonl"):
    j = json.loads(line)
    p = str(j["pmid"])
    t = (j.get("title") or "").strip(); a = (j.get("abstract") or "").strip()
    texts_len[p] = len((t + "\n" + a).strip() if a else t)

def mcnemar(a_hits, c_hits):
    b = sum(1 for x, y in zip(a_hits, c_hits) if x and not y)
    c2 = sum(1 for x, y in zip(a_hits, c_hits) if y and not x)
    nn = b + c2
    p = 1.0 if nn == 0 else min(1.0, 2 * sum(math.comb(nn, i) for i in range(min(b, c2) + 1)) / 2 ** nn)
    return b, c2, p

short_targets = [p for p in rd["dense_en (Qwen3-8B, 英文直查)"]["rank_details"] if texts_len.get(p, 0) <= 2000]
medcpt_ranks_src = json.load(open(f"{LT}/n9_medcpt_recall.json"))["rank_details"]
rank_src = {
    "dense_en (Qwen3-8B, 英文直查)": rd["dense_en (Qwen3-8B, 英文直查)"]["rank_details"],
    "medcpt_768d (领域专训对照)": {p: r for p, r in medcpt_ranks_src.items() if p != VOID},
}
rob = {"subset": "目标文献 ≤2000 字符（MedCPT 512-token 预算内）", "n": len(short_targets)}
for arm_name in ["dense_en (Qwen3-8B, 英文直查)", "medcpt_768d (领域专训对照)"]:
    r = [rank_src[arm_name][p] for p in short_targets]
    rob[arm_name.split(" ")[0]] = {"@1": f"{sum(x<=1 for x in r)}/{len(r)}",
                                   "@5": f"{sum(x<=5 for x in r)}/{len(r)}",
                                   "@100": f"{sum(x<=100 for x in r)}/{len(r)}"}
b1, c1_, p1_ = mcnemar([rank_src["dense_en (Qwen3-8B, 英文直查)"][p] <= 1 for p in short_targets],
                       [rank_src["medcpt_768d (领域专训对照)"][p] <= 1 for p in short_targets])
b100, c100_, p100_ = mcnemar([rank_src["dense_en (Qwen3-8B, 英文直查)"][p] <= 100 for p in short_targets],
                             [rank_src["medcpt_768d (领域专训对照)"][p] <= 100 for p in short_targets])
rob["mcnemar"] = {"@1": f"b={b1} c={c1_} p={p1_:.3f}", "@100": f"b={b100} c={c100_} p={p100_:.3f}"}
rob["verdict"] = ("差距保持（变量被排除）" if (p1_ < 0.05 or p100_ < 0.05) else "差距不再显著（预算变量不可排除，需披露）")
d["_robustness_short_docs"] = rob

json.dump(d, open(f"{LT}/n9_final_table_v2.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(rob, ensure_ascii=False, indent=1))
print("table updated")
