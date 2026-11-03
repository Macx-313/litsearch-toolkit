import os
#!/usr/bin/env python3
# PMID→字节偏移索引：扫 full_corpus.jsonl 一遍，建 SQLite（pmid, offset），供毫秒级单条取摘要
import json, os, sqlite3, sys, time

LT = os.environ.get("LITSEARCH_HOME", os.path.expanduser("~/litsearch"))
SRC = f"{LT}/parsed/full_corpus.jsonl"
DB = f"{LT}/parsed/pmid_offset.db"

def build():
    if os.path.exists(DB):
        os.remove(DB)
    con = sqlite3.connect(DB)
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("CREATE TABLE loc (pmid INTEGER PRIMARY KEY, off INTEGER)")
    t0 = time.time()
    n = 0
    batch = []
    with open(SRC, "rb") as f:
        while True:
            off = f.tell()
            line = f.readline()
            if not line:
                break
            d = json.loads(line)
            batch.append((d["pmid"], off))
            n += 1
            if len(batch) >= 200000:
                con.executemany("INSERT OR REPLACE INTO loc VALUES (?,?)", batch)
                batch = []
                if n % 5000000 < 200000:
                    print(f"  {n} rows ({time.time()-t0:.0f}s)", flush=True)
    if batch:
        con.executemany("INSERT OR REPLACE INTO loc VALUES (?,?)", batch)
    con.commit()
    cnt = con.execute("SELECT COUNT(*) FROM loc").fetchone()[0]
    con.close()
    print(f"DONE rows={cnt} db_size={os.path.getsize(DB)/1e9:.2f}GB elapsed={time.time()-t0:.0f}s", flush=True)

def lookup(pmids):
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    out = []
    with open(SRC, "rb") as f:
        for p in pmids:
            row = con.execute("SELECT off FROM loc WHERE pmid=?", (int(p),)).fetchone()
            if not row:
                out.append(None); continue
            f.seek(row[0])
            d = json.loads(f.readline())
            out.append(d)
    con.close()
    return out

if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    elif sys.argv[1] == "lookup":
        import time as _t
        t0 = _t.time()
        for d in lookup(sys.argv[2:]):
            print(json.dumps({k: (v[:80] + "…" if isinstance(v, str) and len(v) > 80 else v) for k, v in d.items()}, ensure_ascii=False) if d else "NOT FOUND")
        print(f"[lookup {len(sys.argv)-2} 条耗时 {(_t.time()-t0)*1000:.0f}ms]", file=sys.stderr)
