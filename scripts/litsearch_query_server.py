#!/usr/bin/env python3
# 文献库查询编码服务（:8004）——与语料同模型 Qwen3-Embedding-8B（N-11：双侧同 model_id+dim）
# 接口：POST /embed {"texts": ["..."]} -> {"vectors": [[...]], "dim": N}
import os
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from sentence_transformers import SentenceTransformer

MODEL = None
DIM = 0
MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
INSTR = "Instruct: Given a biomedical research question, retrieve relevant English literature abstracts\nQuery: "

def get_model():
    global MODEL, DIM
    if MODEL is None:
        print("[litq] loading " + MODEL_NAME + " (CPU, fp32)...", flush=True)
        MODEL = SentenceTransformer(MODEL_NAME, device="cpu")
        MODEL.max_seq_length = 512
        DIM = MODEL.get_embedding_dimension()
        print(f"[litq] ready dim={DIM} model={MODEL_NAME}", flush=True)
    return MODEL

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": MODEL is not None, "dim": DIM,
                                         "model": MODEL_NAME}).encode())
        else:
            self.send_response(404); self.end_headers()
    def do_POST(self):
        if self.path != "/embed":
            self.send_response(404); self.end_headers(); return
        n = int(self.headers.get("Content-Length", 0))
        d = json.loads(self.rfile.read(n))
        # 查询侧带 instruction（对称侧免）；单一模型无文档接口误用时也安全
        texts = [t if t.startswith("Instruct:") else INSTR + t for t in d.get("texts", [])]
        m = get_model()
        v = m.encode(texts, batch_size=8, normalize_embeddings=True,
                     convert_to_numpy=True).astype("float32")
        out = json.dumps({"vectors": v.tolist(), "dim": DIM}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(out)
    def log_message(self, *a):
        pass

if __name__ == "__main__":
    get_model()  # 启动即加载
    ThreadingHTTPServer(("0.0.0.0", 8004), H).serve_forever()
