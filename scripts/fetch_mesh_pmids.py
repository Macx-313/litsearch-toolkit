#!/usr/bin/env python3
# 拉取 10 个 MeSH 词（子树扩展语义，与 esearch 一致）的全量 PMID 清单
# 用 webenv/history 分页 efetch，落盘 p1_mesh_pmids.txt；Count 交叉核对
import json, time, urllib.request, urllib.parse, xml.etree.ElementTree as ET

QUERY = '"Stroke"[Mesh] OR "Intracranial Hemorrhages"[Mesh] OR "Subarachnoid Hemorrhage"[Mesh] OR "Cerebral Infarction"[Mesh] OR "Brain Ischemia"[Mesh] OR "Brain Injuries, Traumatic"[Mesh] OR "Craniocerebral Trauma"[Mesh] OR "Intracranial Pressure"[Mesh] OR "Coma"[Mesh] OR "Status Epilepticus"[Mesh]'
BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

def get(url):
    for i in range(5):
        try:
            return urllib.request.urlopen(url, timeout=60).read().decode()
        except Exception as e:
            print("retry", i, e); time.sleep(3)
    raise RuntimeError("fetch failed " + url)

# 1) esearch 拿 Count + WebEnv
p = urllib.parse.urlencode({"db": "pubmed", "term": QUERY, "retmax": 0, "usehistory": "y", "retmode": "json"})
j = json.loads(get(BASE + "esearch.fcgi?" + p))["esearchresult"]
count, webenv, qk = int(j["count"]), j["webenv"], j["querykey"]
print(f"esearch Count={count} (历史参照 541,894=2026-09-25 实测勿引用; 2026-09-26 实测 541,970)")

# 2) efetch 分页拉 PMID（200/req 上限？efetch retmax 上限 10000）
pmids = []
step = 10000
for start in range(0, count, step):
    p = urllib.parse.urlencode({"db": "pubmed", "WebEnv": webenv, "query_key": qk,
                                "retstart": start, "retmax": step, "rettype": "uilist", "retmode": "xml"})
    xml = get(BASE + "efetch.fcgi?" + p)
    ids = [e.text for e in ET.fromstring(xml).iter("Id")]
    pmids.extend(ids)
    if start % 100000 == 0:
        print(f"  fetched {len(pmids)}/{count}", flush=True)
    time.sleep(0.4)
print(f"total fetched={len(pmids)} unique={len(set(pmids))}")
open(f"{LT}/parsed/p1_mesh_pmids.txt", "w").write("\n".join(pmids))
print("saved -> parsed/p1_mesh_pmids.txt")
