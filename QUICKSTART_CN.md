# 快速开始（中文版）

## 这是什么
一套完整管线，让你在自有硬件上构建 2830 万篇 PubMed 文献的本地语义检索系统。
中文自然语言提问 → 4 秒返回 PMID + 标题 + 相关度分。

## 硬件要求
- 编码阶段：2× GPU（48GB+ VRAM/卡），约 4.5 天
- 服务阶段：150GB+ RAM（量化索引 58GB + 页缓存），无 GPU 也可（CPU 编码 1-4s/查询）
- 磁盘：≥2TB NVMe（原始数据 51GB + 语料 40GB + 向量 432GiB + Qdrant ~570GB）

## 数据前提
PubMed baseline 是 NLM 官方免费数据（需署名、不得暗示 NLM 背书）。
本工具包**不含**任何数据——你需要自行从官方下载。

## 五步建库
1. 下载 PubMed baseline（51GB，1334 个文件）→ `pubmed/baseline/`
2. 解析 → `parse_all_runner.sh`（16 路并行，~20 分钟）
3. 编码 → `litsearch_b_window.sh "bash master_p2.sh"`（DP2 双卡，~4.5 天）
4. 建库 → `build_p2.py`（含强制对账，~9 小时 gRPC）
5. 部署查询 → `litsearch_query_server.py`(:8004) + `litsearch.sh`

## 典型查询
```bash
bash litsearch.sh "急性硬膜下血肿快速自发缩小的病例报道" 20
bash litsearch.sh "machine learning mortality prediction intracerebral hemorrhage" 50
```

## 已实测性能
| 指标 | 值 |
|---|---|
| 已知项召回 @100 | 98.1%（同口径子集库 100%） |
| vs PubMed 关键词检索 | 90.3% vs 51.6%（p<0.001） |
| 延迟 p50 | 281ms（默认）/ 32ms（快档） |
| 端到端单查询 | ~4s（含 CPU 编码） |
