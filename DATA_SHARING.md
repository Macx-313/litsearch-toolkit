# 向量数据库分发说明

## 分发内容（两档任选）

### 档 A：Qdrant 即用索引（推荐大多数人）
| 文件 | 大小 | 说明 |
|---|---|---|
| `qdrant_storage/` 整目录 | ~569GB | 已优化 GREEN 的完整索引（pq4+HNSW+payload），起容器即查 |

**接收方只需**：
```bash
docker run -d -p 6333:6333 -v /path/to/qdrant_storage:/qdrant/storage qdrant/qdrant:v1.19.1
# 立即可查（需另配查询编码服务 :8004）
```

### 档 B：原始 fp32 向量（进阶用户）
| 文件 | 大小 | 说明 |
|---|---|---|
| `p2_half0.npy` + `p2_half0.pmids.txt` | ~216GiB + 14MB | 前半 14,168,536 篇 |
| `p2_half1.npy` + `p2_half1.pmids.txt` | ~216GiB + 14MB | 后半 14,168,536 篇 |

**接收方需**：跑 `build_p2.py`（~9h 纯 CPU），可自选量化方案 / 换 Qdrant 或其他向量库

### 两档均需（文本层，自行生成）
接收方需自行从 NLM 官方下载 PubMed baseline（~51GB），用 `parse_all.py`（~20 min）生成 `full_corpus.jsonl`，再跑 `build_offset_index.py`（~3 min）建 PMID→偏移索引——这样**检索结果的标题/摘要才能展示**。文本层不打包分发（含出版商版权内容，各自私用）。

## 法律依据

1. **PubMed 数据**：NLM 官方允许自由再分发（署名 + 不用词标），baseline 任何人可从官方下载
2. **嵌入向量 ≠ 文本**：数学变换，不可逆向还原原文
3. **直接先例**：AllenAI 以 Apache-2.0 公开分发 SPECTER2 嵌入（120M 篇、~840GB），涵盖所有出版商摘要，多年无争议
4. **Qwen3-Embedding-8B**：Apache-2.0 开源模型

## 节省的算力

| 自建 | 接收档 A | 接收档 B |
|---|---|---|
| 2× GPU（48GB+）× 4.5 天 + 9h CPU 建库 + 40h 索引优化 | **零计算**，起容器即查 | 9h CPU 建库 + ~40h 索引优化 |
| ≈ 216 GPU·h + 49h CPU | ≈ **0** | ≈ **49h CPU** |

## 接收方完整部署（档 A 为例）

```bash
# 1. 下载 qdrant_storage/（~569GB）
# 2. 下载 PubMed baseline 生成文本层（~51GB + 23 min）
wget -r https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/  # 或镜像
python3 scripts/parse_all_runner.sh                     # 生成 full_corpus.jsonl
python3 scripts/build_offset_index.py build             # 生成 pmid_offset.db

# 3. 起服务
docker run -d -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant:v1.19.1
python3 scripts/litsearch_query_server.py &             # :8004 编码服务

# 4. 查询
bash litsearch.sh "急性硬膜下血肿快速自发缩小" 20
