import os
#!/usr/bin/env python3
# 合并 16 分片子集：PMID 去重（保留首现）+ 汇总 N-5 统计 + 抽样校验
import json, glob, sys

LT = os.environ.get("LITSEARCH_HOME", os.path.expanduser("~/litsearch"))
seen = set()
out = open(f"{LT}/parsed/p1_neurocrit.jsonl", "w", encoding="utf-8")
n = 0
dup = 0
for fp in sorted(glob.glob(f"{LT}/parsed/sub_*.jsonl")):
    with open(fp, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            p = d.get("pmid")
            if p in seen:
                dup += 1
                continue
            seen.add(p)
            out.write(line)
            n += 1
out.close()
print(f"subset_unique={n} duplicates_removed={dup}")

# N-5 统计汇总
agg = {"records": 0, "records_with_abstract": 0, "records_no_mesh": 0, "mesh_hit": 0,
       "total_abstract_chars": 0, "files_parsed": 0, "no_mesh_pubtypes": {}, "no_mesh_years": {}}
for fp in sorted(glob.glob(f"{LT}/parsed/stat_*.json")):
    s = json.load(open(fp))
    for k in ["records", "records_with_abstract", "records_no_mesh", "mesh_hit",
              "total_abstract_chars", "files_parsed"]:
        agg[k] += s.get(k, 0)
    for k in ["no_mesh_pubtypes", "no_mesh_years"]:
        for kk, vv in s.get(k, {}).items():
            agg[k][kk] = agg[k].get(kk, 0) + vv
agg["no_mesh_share_pct"] = round(100 * agg["records_no_mesh"] / max(agg["records"], 1), 2)
agg["abstract_coverage_pct"] = round(100 * agg["records_with_abstract"] / max(agg["records"], 1), 2)
agg["avg_abstract_chars"] = agg["total_abstract_chars"] // max(agg["records_with_abstract"], 1)
top_pt = sorted(agg["no_mesh_pubtypes"].items(), key=lambda x: -x[1])[:6]
top_yr = sorted(agg["no_mesh_years"].items(), key=lambda x: -x[1])[:6]
json.dump(agg, open(f"{LT}/parsed/n5_stats.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in agg.items() if not isinstance(v, dict)}, ensure_ascii=False))
print("top_no_mesh_pubtypes:", top_pt)
print("top_no_mesh_years:", top_yr)

# 子集内摘要覆盖
n_abs = 0
with open(f"{LT}/parsed/p1_neurocrit.jsonl", encoding="utf-8") as f:
    for line in f:
        if json.loads(line).get("abstract"):
            n_abs += 1
print(f"subset_abstract_coverage={n_abs}/{n} ({100*n_abs//max(n,1)}%)")
