# query_rag.py
# Mô tả: file này thực hiện RAG-style search:
# 1) Tạo embedding cho query bằng BGE-M3
# 2) Tìm top-k nearest neighbors từ FAISS index
# 3) Lấy văn bản tương ứng từ SQLite DB
# 4) Rerank các cặp (query, doc) bằng BGE-Reranker (CrossEncoder)
# 5) Trả về top-k kết quả có score cao nhất

import sqlite3      # Làm việc với SQLite (lưu văn bản + metadata)
import json         # Xử lý metadata JSON trong DB
import numpy as np  # Toán ma trận, chuẩn hoá vector
import faiss        # FAISS: tìm kiếm vector nhanh (Facebook AI)
import torch        # PyTorch: chạy model
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
# AutoTokenizer, AutoModel để load BGE-M3 (embedding)
# AutoModelForSequenceClassification để load BGE-Reranker

# ---------------------------
# Cấu hình hằng số
# ---------------------------
BGE_MODEL = "src/notebook/Data_Fine_Tune/BAAI/bge-m3"                # Model embedding
RERANK_MODEL = "BAAI/bge-reranker-v2-m3" # Model reranker
MAX_LENGTH_EMBED = 512                   # Giới hạn token khi embed
MAX_LENGTH_RERANK = 512                  # Giới hạn token khi rerank
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Sử dụng GPU nếu có, ngược lại dùng CPU

# Đường dẫn cơ sở dữ liệu & FAISS index
FAISS_FILE = "./database/laws_bge.index"
VECTOR_DB = "./database/laws_bge.db"

# ---------------------------
# Load models
# ---------------------------
print("Đang load BGE-M3 (embedding) ...")
tokenizer_bge = AutoTokenizer.from_pretrained(BGE_MODEL)         # Load tokenizer cho BGE-M3
model_bge = AutoModel.from_pretrained(BGE_MODEL).to(DEVICE).eval() # Load model embedding, đặt chế độ eval

print("Đang load BGE-Reranker ...")
tokenizer_rerank = AutoTokenizer.from_pretrained(RERANK_MODEL)   # Tokenizer cho reranker
model_rerank = AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL).to(DEVICE).eval()
# Load model reranker, để chế độ eval

# ---------------------------
# Hàm tạo embedding cho query
# ---------------------------
@torch.no_grad()  # Tắt gradient, tiết kiệm bộ nhớ khi inference
def embed_query(query: str) -> np.ndarray:
    """
    Nhận query string, trả về embedding vector (1, hidden_size)
    Sử dụng mean pooling + attention mask, chuẩn hoá L2.
    """
    if not query or not query.strip():   # Nếu query rỗng -> trả về vector 0
        return np.zeros((1, model_bge.config.hidden_size), dtype="float32")

    # Tokenize query (chuyển sang tensor)
    enc = tokenizer_bge(
        query,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH_EMBED,
        return_tensors="pt"
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()} # Đưa tensor lên GPU/CPU

    out = model_bge(**enc)                   # Chạy model
    last_hidden = out.last_hidden_state      # Lấy hidden states (batch, seq_len, hidden_size)

    # Tạo mask từ attention_mask để bỏ padding
    mask = enc["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()

    summed = torch.sum(last_hidden * mask, dim=1) # Tổng vector (chỉ token thật)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9) # Đếm số token thật
    mean = summed / counts                        # Lấy trung bình -> embedding

    arr = mean.cpu().numpy()                      # Chuyển sang numpy
    arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9) # Chuẩn hoá L2
    return arr.astype("float32")                  # Trả về vector float32

# ---------------------------
# Hàm load FAISS index
# ---------------------------
def load_index(faiss_file: str):
    try:
        idx = faiss.read_index(faiss_file)     # Đọc index từ file
        print(f"FAISS index loaded: {faiss_file}")
        return idx
    except Exception as e:
        raise RuntimeError(f"Không thể load FAISS index '{faiss_file}': {e}")

# ---------------------------
# Lấy document từ SQLite theo faiss_id
# ---------------------------
def fetch_docs_by_faiss_ids(conn, faiss_ids):
    if not faiss_ids:
        return []

    # Tạo list dấu ? (?,?,...) tương ứng với số id
    q_marks = ",".join("?" * len(faiss_ids))

    # Truy vấn DB lấy các field
    rows = conn.execute(
        f"SELECT faiss_id, uuid, doc_text, metadata_json FROM vectors WHERE faiss_id IN ({q_marks})",
        tuple(faiss_ids)
    ).fetchall()

    # Chuyển sang dict {faiss_id: {...}}
    rows_map = {
        r[0]: {"uuid": r[1], "text": r[2], "metadata": json.loads(r[3]) if r[3] else {}}
        for r in rows
    }
    # Trả về list theo đúng thứ tự ids
    return [rows_map[i] for i in faiss_ids if i in rows_map]

# ---------------------------
# Hàm rerank bằng BGE-Reranker
# ---------------------------
@torch.no_grad()
def rerank(query: str, docs: list, batch_size: int = 16):
    if not docs:
        return []

    scores = []  # list chứa điểm rerank
    texts = [d["text"] for d in docs]  # Lấy text của từng doc

    # Chia batch để rerank (tránh OOM GPU)
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        # Tokenize cặp (query, doc)
        inputs = tokenizer_rerank(
            [query] * len(batch_texts), # query nhân lên cho bằng số doc
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH_RERANK,
            return_tensors="pt"
        ).to(DEVICE)

        out = model_rerank(**inputs)       # Chạy reranker
        batch_scores = out.logits.squeeze().tolist() # Lấy logits
        if isinstance(batch_scores, float):  # Nếu chỉ có 1 doc -> float
            batch_scores = [batch_scores]
        scores.extend(batch_scores)

    # Ghép docs với scores và sắp xếp giảm dần
    ranked_docs = [d for _, d in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
    return ranked_docs

# ---------------------------
# Hàm chính: RAG search
# ---------------------------
def rag_search(query: str, top_k=30, top_k_rerank=5):
    conn = sqlite3.connect(VECTOR_DB)     # Mở kết nối DB
    index = load_index(FAISS_FILE)        # Load FAISS index

    q_emb = embed_query(query)            # Embed query
    D, I = index.search(q_emb, top_k) # FAISS search (trả distances D và indices I)

    ids = I[0].tolist()                   # Chuyển ids thành list
    docs = fetch_docs_by_faiss_ids(conn, ids) # Lấy docs từ DB
    conn.close()

    reranked_docs = rerank(query, docs)   # Rerank docs

    results = []
    for i, doc in enumerate(reranked_docs[:top_k_rerank]): # Lấy top-k rerank
        item = {
            "score": float(D[0][i]),      # Lấy FAISS score (không phải reranker score)
            "uuid": doc["uuid"],
            "text": doc["text"],
            "metadata": doc["metadata"]
        }
        results.append(item)
    return results

# ---------------------------
# Ví dụ chạy trực tiếp file
# ---------------------------
if __name__ == "__main__":
    query = "Khoản 1 Điều 8 luật hôn nhân"
    try:
        results = rag_search(query, faiss_top_k=30, rerank_top_k=5)
        for i, r in enumerate(results):
            print(f"Rank {i+1} | uuid={r['uuid']} | score={r['score']:.4f}")
            print(r['text'][:300], "...")   # In 300 ký tự đầu tiên
            print("metadata:", r['metadata'])
            print("----")
    except Exception as e:
        print("Lỗi khi chạy rag_search:", e)
