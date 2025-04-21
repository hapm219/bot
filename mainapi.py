from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pathlib import Path
import glob, json, time, base64
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from load_model import load_llm

DATA_DIR = Path("data")
INDEX_DIR = DATA_DIR / "index"
EMBEDDING_PATH = DATA_DIR / "encoder" / "bge-m3"
MAX_TOKENS = 192
TOP_K = 3

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

class ChatRequest(BaseModel):
    question: str

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

def search_similar_chunks(query: str, top_k=TOP_K):
    query_emb = embed_model.encode(
        f"Represent this sentence for searching relevant passages: {query}",
        convert_to_numpy=True
    )
    results = []
    for index, texts in zip(all_indexes, all_mappings):
        D, I = index.search(np.array([query_emb]), top_k)
        results += [(score, texts[idx]) for score, idx in zip(D[0], I[0]) if 0 <= idx < len(texts)]
    return [text for _, text in sorted(results, key=lambda x: x[0])[:top_k]]

def limit_context(chunks, max_chars=800):
    context = ""
    for c in chunks:
        if len(context) + len(c) > max_chars:
            break
        context += c + "\n\n"
    return context.strip()

@app.post("/chat")
async def chat(req: ChatRequest):
    q = req.question.strip()
    top_chunks = search_similar_chunks(q)
    context = limit_context(top_chunks)

    prompt = f"""Bạn là trợ lý AI IDCee. Trả lời ngắn gọn, bằng tiếng Việt và chỉ dựa vào phần Thông tin nội bộ bên dưới.

❗ Không được suy đoán, không được bịa.
❗ Nếu trong văn bản có ghi cụ thể (số liệu, thời gian, người chịu trách nhiệm, định kỳ...), phải trả lời chính xác không thiếu sót.
❗ Nếu không tìm thấy thông tin liên quan, chỉ được trả lời: "Tôi không tìm thấy thông tin trong tài liệu nội bộ để trả lời câu hỏi này."

Câu hỏi: {q}

Thông tin nội bộ:
{context}

Trả lời:
"""

    start_time = time.time()
    token_count = 0

    async def generate():
        nonlocal token_count
        result = llm(prompt, max_tokens=MAX_TOKENS, stream=True)
        if hasattr(result, "__iter__"):
            for chunk in result:
                delta = chunk["choices"][0]["text"] if "choices" in chunk else str(chunk)
                token_count += len(delta.strip().split())
                yield delta
        else:
            yield str(result)

        # ✅ Gửi metadata dạng Base64 cuối stream
        duration = round(time.time() - start_time, 2)
        meta = {
            "response_time_sec": duration,
            "new_tokens": token_count,
            "context_length": len(context),
            "top_k_chunks": top_chunks,
            "context_used": context.strip()
        }
        meta_str = json.dumps(meta, ensure_ascii=False)
        meta_b64 = base64.b64encode(meta_str.encode("utf-8")).decode("ascii")
        yield f"\n[[[META-B64]]]{meta_b64}"

    return StreamingResponse(generate(), media_type="text/plain")
