from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import glob, json, gc
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from load_model import load_llm

# ====== CẤU HÌNH ======
DATA_DIR = Path("data")
INDEX_DIR = DATA_DIR / "index"
EMBEDDING_PATH = DATA_DIR / "encoder" / "bge-m3"
MAX_TOKENS = 192
TOP_K = 3

# ====== KHỞI TẠO FASTAPI ======
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

class ChatRequest(BaseModel):
    question: str

# ====== LOAD MÔ HÌNH & INDEX ======
print("📥 Đang tải mô hình embedding...")
embed_model = SentenceTransformer(str(EMBEDDING_PATH))

print("📦 Đang tải toàn bộ FAISS index...")
all_indexes, all_mappings = [], []
for faiss_path in glob.glob(str(INDEX_DIR / "**/index.faiss"), recursive=True):
    try:
        index = faiss.read_index(faiss_path)
        mapping_path = Path(faiss_path).parent / "mapping.json"
        if not mapping_path.exists():
            continue
        with open(mapping_path, encoding="utf-8") as f:
            mapping = json.load(f)
        all_indexes.append(index)
        all_mappings.append(mapping)
        print(f"✅ Loaded index: {faiss_path}")
    except Exception as e:
        print(f"❌ Lỗi khi load {faiss_path}: {e}")

print("🤖 Đang chuẩn bị mô hình LLM...")
llm = load_llm()

# ====== HÀM XỬ LÝ ======
def search_similar_chunks(query: str, top_k=TOP_K):
    query_emb = embed_model.encode(
        f"Represent this sentence for searching relevant passages: {query}",
        convert_to_numpy=True
    )
    results = []
    for index, texts in zip(all_indexes, all_mappings):
        D, I = index.search(np.array([query_emb]), top_k)
        results += [(score, texts[str(idx)]) for score, idx in zip(D[0], I[0]) if str(idx) in texts]
    return [text for _, text in sorted(results, key=lambda x: x[0])[:top_k]]

def limit_context(chunks, max_chars=800):
    context = ""
    for c in chunks:
        if len(context) + len(c) > max_chars:
            break
        context += c + "\n\n"
    return context.strip()

# ====== ENDPOINT ======
@app.post("/chat")
async def chat(req: ChatRequest):
    q = req.question.strip()
    chunks = search_similar_chunks(q)
    context = limit_context(chunks)

    prompt = f"""Bạn là trợ lý AI IDCee. Trả lời ngắn gọn, bằng tiếng Việt và chỉ sử dụng thông tin từ phần Thông tin nội bộ.

👉 Nếu không có thông tin phù hợp, trả lời duy nhất câu sau:
"Tôi không tìm thấy thông tin trong tài liệu nội bộ để trả lời câu hỏi này."

❗ Không được bịa, suy đoán hoặc thêm nội dung ngoài dữ liệu cung cấp. Chỉ sử dụng nội dung chứa từ khóa: "{q}"

### Câu hỏi:
{q}

### Thông tin nội bộ:
{context}

### Trả lời:
"""

    try:
        raw = llm(prompt, max_tokens=MAX_TOKENS)
        if isinstance(raw, str):
            return {"answer": raw.strip()}
        elif hasattr(raw, "__iter__"):
            return {"answer": "".join(chunk for chunk in raw).strip()}
        elif isinstance(raw, dict) and "choices" in raw:
            return {"answer": raw["choices"][0]["text"].strip()}
        else:
            return {"answer": "(Không thể xử lý phản hồi)"}
    except Exception as e:
        return {"answer": f"❌ Lỗi infer: {e}"}
