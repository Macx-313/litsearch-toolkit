import os
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B 段语料编码器（vLLM offline，GPU 窗口内运行）
用法: python3 encode_corpus.py <in.jsonl> <out_prefix> [--model Qwen/Qwen3-Embedding-8B] [--quant fp8|none] [--max-len 1024] [--batch 256]
输入 jsonl 需含 pmid 与 title/abstract（或 content）字段
输出: <out_prefix>.npy (float32 向量) + <out_prefix>.pmids.txt + <out_prefix>.report.json
断点: <out_prefix>.progress 记录已完成行号，重跑自动续
token 精确计数（X-3）: report 内 n_tokens 为 tokenizer 实测值
"""
import argparse, json, os, sys, time
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile"); ap.add_argument("out_prefix")
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    ap.add_argument("--quant", default="none", choices=["none", "fp8"])
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--gpu-frac", type=float, default=0.92)
    ap.add_argument("--tp", type=int, default=2)
    args = ap.parse_args()

    from vllm import LLM
    from transformers import AutoTokenizer

    kw = dict(model=args.model, runner="pooling", max_model_len=args.max_len,
              gpu_memory_utilization=args.gpu_frac, enforce_eager=False,
              tensor_parallel_size=args.tp)
    if args.quant == "fp8":
        kw["quantization"] = "fp8"
    llm = LLM(**kw)
    tok = AutoTokenizer.from_pretrained(args.model)

    # 读入（题录级：title + abstract；无 abstract 用 title）
    pmids, texts = [], []
    with open(args.infile, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            t = (d.get("title") or "").strip()
            a = (d.get("abstract") or d.get("content") or "").strip()
            txt = (t + "\n" + a).strip() if a else t
            if not txt:
                continue
            pmids.append(str(d.get("pmid", len(pmids))))
            texts.append(txt)
    n = len(texts)
    print(f"[encode] loaded {n} docs", flush=True)

    # 断点（格式 "encoded:dim"，dim 供续跑重建 memmap）
    prog_path = args.out_prefix + ".progress"
    done, dim_saved = 0, None
    if os.path.exists(prog_path):
        with open(prog_path) as f:
            parts = f.read().strip().split(":")
            done = int(parts[0])
            if len(parts) > 1 and parts[1].isdigit():
                dim_saved = int(parts[1])
    vecs = None
    if done > 0 and dim_saved and os.path.exists(args.out_prefix + ".vecs.f32"):
        vecs = np.memmap(args.out_prefix + ".vecs.f32", dtype=np.float32, mode="r+", shape=(n, dim_saved))
        print(f"[encode] resume from {done} (dim={dim_saved})", flush=True)

    t0 = time.time()
    n_tokens = 0
    encoded = done
    LIMIT = args.max_len - 24  # 预算内截断（超长摘要头部截断，对检索无损）
    pf = open(args.out_prefix + ".pmids.txt", "a", encoding="utf-8")
    CH = max(5000, args.batch * 10)  # 检查点粒度（崩了最多丢几分钟）
    i = done
    while i < n:
        j = min(i + args.batch, n)
        batch = texts[i:j]
        # token 精确计数（X-3，按原文计）+ 超长截断
        enc_ids = tok(batch, add_special_tokens=True)["input_ids"]
        safe = []
        for txt, ids in zip(batch, enc_ids):
            n_tokens += len(ids)
            if len(ids) > LIMIT:
                safe.append(tok.decode(ids[:LIMIT], skip_special_tokens=True))
            else:
                safe.append(txt)
        out = llm.embed(safe, use_tqdm=False)
        dim = len(out[0].outputs.embedding)
        if vecs is None:
            vecs = np.memmap(args.out_prefix + ".vecs.f32", dtype=np.float32, mode="w+", shape=(n, dim))
        for k, o in enumerate(out):
            vecs[i + k] = np.asarray(o.outputs.embedding, dtype=np.float32)
        for p in pmids[i:j]:
            pf.write(p + "\n")
        encoded = j
        if encoded % CH < args.batch or j == n:
            pf.flush()
            with open(prog_path, "w") as f:
                f.write(f"{encoded}:{dim}")
        el = time.time() - t0
        if encoded % 5000 < args.batch:
            print(f"[encode] {encoded}/{n} elapsed={el:.0f}s rate={(encoded-done)/el:.1f}docs/s tok/s={n_tokens/el:.0f}", flush=True)
        i = j
    pf.close()

    # 收尾：memmap → npy（原子拷贝）
    dim = vecs.shape[1]
    arr = np.memmap(args.out_prefix + ".vecs.f32", dtype=np.float32, mode="r", shape=(n, dim))
    np.save(args.out_prefix + ".npy", np.asarray(arr))
    os.remove(args.out_prefix + ".vecs.f32")
    el = time.time() - t0
    report = {"model": args.model, "quant": args.quant, "n_docs": n, "dim": dim,
              "n_tokens_exact": n_tokens, "elapsed_s": round(el, 1),
              "docs_per_s": round((n - done) / el, 2), "tokens_per_s": round(n_tokens / el, 1),
              "ts": time.strftime("%F %T")}
    with open(args.out_prefix + ".report.json", "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("[encode] DONE " + json.dumps(report, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
