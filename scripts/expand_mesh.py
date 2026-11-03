#!/usr/bin/env python3
# MeSH 子树扩展词表生成器（H-8 落盘：从 10 种子词 + mtrees2026.bin 再生 100 词扩展词表）
# 用法: python3 expand_mesh.py <mtrees2026.bin> <out_mesh_expanded.txt>
# 种子词与生成逻辑与 2026-09-26 生成 mesh_expanded.txt 时完全一致（md5 可对 MANIFEST）
import sys

SEEDS = ['Stroke', 'Intracranial Hemorrhages', 'Subarachnoid Hemorrhage', 'Cerebral Infarction',
         'Brain Ischemia', 'Brain Injuries, Traumatic', 'Craniocerebral Trauma',
         'Intracranial Pressure', 'Coma', 'Status Epilepticus']

def main():
    mtrees, out = sys.argv[1], sys.argv[2]
    pairs = []
    for line in open(mtrees):
        line = line.strip()
        if ';' in line:
            name, tn = line.rsplit(';', 1)
            pairs.append((name, tn))
    prefixes = [tn for name, tn in pairs if name in SEEDS]
    expanded = set(SEEDS)
    for name, tn in pairs:
        if any(tn == p or tn.startswith(p + '.') for p in prefixes):
            expanded.add(name)
    open(out, 'w').write('\n'.join(sorted(expanded)))
    print(f"seeds={len(SEEDS)} tree_prefixes={len(prefixes)} expanded={len(expanded)} -> {out}")

if __name__ == "__main__":
    main()
