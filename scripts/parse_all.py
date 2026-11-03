import os
#!/usr/bin/env python3
# P2 全量语料解析：输出全部"有摘要"记录（28.3M 条）为 jsonl，无 MeSH 过滤
# 用法同 parse_baseline.py：<baseline_dir> <out.jsonl> <out_stats.json>
import gzip, glob, json, os, sys
import xml.etree.ElementTree as ET

def main():
    baseline_dir, out_jsonl, out_stats = sys.argv[1], sys.argv[2], sys.argv[3]
    prog_path = out_jsonl + ".progress"
    done_files = set()
    if os.path.exists(prog_path):
        with open(prog_path) as f:
            done_files = set(l.strip() for l in f if l.strip())
    files = sorted(glob.glob(os.path.join(baseline_dir, "pubmed26n*.xml.gz")))
    stats = {"files_total": len(files), "files_parsed": 0, "records": 0, "emitted": 0}
    out = open(out_jsonl, "a", encoding="utf-8")
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
                    abs_parts = []
                    for at in art.findall("Article/Abstract/AbstractText"):
                        label = at.get("Label")
                        txt = "".join(at.itertext()).strip()
                        abs_parts.append(f"{label}: {txt}" if label else txt)
                    abstract = " ".join(abs_parts).strip()
                    if abstract:
                        title_el = art.find("Article/ArticleTitle")
                        title = "".join(title_el.itertext()) if title_el is not None else ""
                        year = None
                        for pd in art.findall("Article/Journal/JournalIssue/PubDate/Year"):
                            if pd.text and pd.text.isdigit():
                                year = int(pd.text); break
                        out.write(json.dumps({"pmid": pmid, "title": title, "abstract": abstract,
                                              "year": year}, ensure_ascii=False) + "\n")
                        stats["emitted"] += 1
                    stats["records"] += 1
                    el.clear()
        except (EOFError, OSError, ET.ParseError):
            continue
        stats["files_parsed"] += 1
        prog.write(os.path.basename(fp) + "\n"); prog.flush(); out.flush()
    out.close(); prog.close()
    with open(out_stats, "w") as f:
        json.dump(stats, f)
    print(json.dumps(stats))

if __name__ == "__main__":
    main()
