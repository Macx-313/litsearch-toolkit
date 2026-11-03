import os
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PubMed baseline XML.gz 流式解析 + MeSH 过滤 + N-5 统计
用法: python3 parse_baseline.py <baseline_dir> <out_subset.jsonl> <out_stats.json> [--mesh-file mesh_terms.txt]
  --mesh-file: 每行一个 MeSH Descriptor 名（默认内置 10 核心词）
输出:
  out_subset.jsonl: 命中 MeSH 词的记录 {"pmid","title","abstract","mesh","year","pubtypes","journal"}
  out_stats.json:   N-5 统计（无 MeSH 占比、pubtype/年份分布、总记录数、命中数）
断点续跑: 跳过 out_subset.jsonl.progress 里已处理的文件
"""
import gzip, glob, json, os, re, sys
import xml.etree.ElementTree as ET

DEFAULT_MESH = [
    "Stroke", "Intracranial Hemorrhages", "Subarachnoid Hemorrhage",
    "Cerebral Infarction", "Brain Ischemia", "Brain Injuries, Traumatic",
    "Craniocerebral Trauma", "Intracranial Pressure", "Coma", "Status Epilepticus",
]

def main():
    baseline_dir, out_subset, out_stats = sys.argv[1], sys.argv[2], sys.argv[3]
    mesh_terms = DEFAULT_MESH
    if "--mesh-file" in sys.argv:
        with open(sys.argv[sys.argv.index("--mesh-file")+1]) as f:
            mesh_terms = [l.strip() for l in f if l.strip()]
    mesh_set = set(mesh_terms)

    prog_path = out_subset + ".progress"
    done_files = set()
    if os.path.exists(prog_path):
        with open(prog_path) as f:
            done_files = set(l.strip() for l in f if l.strip())

    files = sorted(glob.glob(os.path.join(baseline_dir, "pubmed26n*.xml.gz")))
    stats = {"files_total": len(files), "files_parsed": 0, "records": 0, "records_with_abstract": 0,
             "records_no_mesh": 0, "mesh_hit": 0, "no_mesh_pubtypes": {}, "no_mesh_years": {},
             "total_abstract_chars": 0}
    out = open(out_subset, "a", encoding="utf-8")
    prog = open(prog_path, "a", encoding="utf-8")

    for fp in files:
        if os.path.basename(fp) in done_files:
            continue
        try:
            with gzip.open(fp, "rb") as gz:
                for _, el in ET.iterparse(gz, events=("end",)):
                    if el.tag != "PubmedArticle":
                        continue
                    art = el.find("MedlineCitation")
                    if art is None:
                        el.clear(); continue
                    pmid_el = art.find("PMID")
                    pmid = int(pmid_el.text) if pmid_el is not None and pmid_el.text else None
                    title_el = art.find("Article/ArticleTitle")
                    title = "".join(title_el.itertext()) if title_el is not None else ""
                    abs_parts = []
                    for at in art.findall("Article/Abstract/AbstractText"):
                        label = at.get("Label")
                        txt = "".join(at.itertext()).strip()
                        if label:
                            abs_parts.append(f"{label}: {txt}")
                        else:
                            abs_parts.append(txt)
                    abstract = " ".join(abs_parts).strip()
                    mesh_list = [d.text for d in art.findall("MeshHeadingList/MeshHeading/DescriptorName") if d.text]
                    year = None
                    for pd in art.findall("Article/Journal/JournalIssue/PubDate/Year"):
                        if pd.text and pd.text.isdigit(): year = int(pd.text); break
                    pubtypes = [p.text for p in art.findall("Article/PublicationList/PublicationType") if p.text]
                    journal_el = art.find("Article/Journal/Title")
                    journal = journal_el.text if journal_el is not None and journal_el.text else ""

                    stats["records"] += 1
                    if abstract: stats["records_with_abstract"] += 1
                    stats["total_abstract_chars"] += len(abstract)
                    if not mesh_list:
                        stats["records_no_mesh"] += 1
                        for pt in pubtypes:
                            stats["no_mesh_pubtypes"][pt] = stats["no_mesh_pubtypes"].get(pt, 0) + 1
                        yk = str(year) if year else "unknown"
                        stats["no_mesh_years"][yk] = stats["no_mesh_years"].get(yk, 0) + 1
                    if mesh_set & set(mesh_list):
                        stats["mesh_hit"] += 1
                        out.write(json.dumps({"pmid": pmid, "title": title, "abstract": abstract,
                                              "mesh": mesh_list, "year": year, "pubtypes": pubtypes,
                                              "journal": journal}, ensure_ascii=False) + "\n")
                    el.clear()
        except (EOFError, OSError, ET.ParseError) as e:
            # 在途/半截文件（并行下载中）或坏档：跳过，不记 progress，下轮重试
            print(f"[skip] {os.path.basename(fp)}: {type(e).__name__}（可能仍在下载）", flush=True)
            continue
        stats["files_parsed"] += 1
        prog.write(os.path.basename(fp) + "\n"); prog.flush()
        out.flush()
    out.close(); prog.close()
    with open(out_stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in stats.items() if not isinstance(v, dict)}, ensure_ascii=False))

if __name__ == "__main__":
    main()
