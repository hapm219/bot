import streamlit as st
import time
from load_model import load_llm
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from pathlib import Path
import glob

TOP_K = 3
INDEX_DIR = Path("data/index")
EMBEDDING_PATH = "data/encoder/bge-m3"

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_PATH, device='cpu')

@st.cache_resource
def load_all_indexes():
    all_indexes = []
    all_mappings = []
    for faiss_path in glob.glob(str(INDEX_DIR / "**/index.faiss"), recursive=True):
        try:
            index = faiss.read_index(faiss_path)
            mapping_file = Path(faiss_path).parent / "mapping.json"
            if not mapping_file.exists():
                continue
            with open(mapping_file, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            all_indexes.append(index)
            all_mappings.append(mapping)
        except Exception as e:
            st.error(f"Lỗi khi load {faiss_path}: {e}")
    return all_indexes, all_mappings

def search_similar_chunks(query, model, all_indexes, all_mappings, top_k=TOP_K):
    query_emb = model.encode(
        f"Represent this sentence for searching relevant passages: {query}",
        convert_to_numpy=True
    )
    results = []
    for index, texts in zip(all_indexes, all_mappings):
        D, I = index.search(np.array([query_emb]), top_k)
        for score, idx in zip(D[0], I[0]):
            if 0 <= idx < len(texts):
                results.append((score, texts[idx]))
    results = sorted(results, key=lambda x: x[0])[:top_k]
    return [r[1] for r in results]

def limit_context(chunks, max_chars=800):
    context = ""
    for c in chunks:
        if len(context) + len(c) > max_chars:
            break
        context += c + "\n\n"
    return context.strip()

# ========= STREAMLIT APP ========= #
st.title("🤖 IDCee - RAG")

query = st.text_input("Nhập câu hỏi:")

if "llm" not in st.session_state:
    with st.spinner("🔁 Đang khởi động mô hình..."):
        st.session_state.llm = load_llm()
        st.session_state.embedding = load_embedding_model()
        st.session_state.indexes, st.session_state.mappings = load_all_indexes()
        st.success("✅ Sẵn sàng!")

if query:
    with st.spinner("🧠 Đang tìm câu trả lời..."):
        start = time.time()
        chunks = search_similar_chunks(
            query,
            st.session_state.embedding,
            st.session_state.indexes,
            st.session_state.mappings
        )
        context = limit_context(chunks)

        prompt = f"""Bạn là trợ lý AI IDCee. Trả lời ngắn gọn, bằng tiếng Việt và chỉ dựa vào phần Thông tin nội bộ bên dưới.

❗ Không được suy đoán, không được bịa.
❗ Nếu trong văn bản có ghi cụ thể (số liệu, thời gian, người chịu trách nhiệm, định kỳ...), phải trả lời chính xác không thiếu sót.
❗ Nếu không tìm thấy thông tin liên quan, chỉ được trả lời: "Tôi không tìm thấy thông tin trong tài liệu nội bộ để trả lời câu hỏi này."

Câu hỏi: {query}

Thông tin nội bộ:
{context}

Trả lời:
"""

        try:
            response = ""
            display = st.empty()

            result = st.session_state.llm(prompt, max_tokens=192, stream=True)

            # ✅ Xử lý đúng định dạng dict từ stream
            if hasattr(result, "__iter__") and not isinstance(result, dict):
                for chunk in result:
                    if isinstance(chunk, dict) and "choices" in chunk:
                        delta = chunk["choices"][0]["text"]
                    else:
                        delta = str(chunk)
                    response += delta
                    display.markdown(f"#### 🤖 IDCee trả lời:\n\n{response.strip() + '▌'}")
                display.markdown(f"#### 🤖 IDCee trả lời:\n\n{response.strip()}")

            elif isinstance(result, str):
                display.markdown(f"#### 🤖 IDCee trả lời:\n\n{result.strip()}")

            elif isinstance(result, dict) and "choices" in result:
                response = result["choices"][0]["text"].strip()
                display.markdown(f"#### 🤖 IDCee trả lời:\n\n{response}")

            else:
                display.markdown("🤖 IDCee: (Không thể xử lý phản hồi)")

            elapsed = time.time() - start
            st.caption(f"⏱️ Thời gian phản hồi: {elapsed:.2f} giây")

            with st.expander("🟨 Đoạn context được sử dụng"):
                st.code(context, language="markdown")

            with st.expander("📄 Top K đoạn truy xuất từ FAISS"):
                for i, chunk in enumerate(chunks):
                    st.markdown(f"**{i+1}.** {chunk.strip()}")

        except Exception as e:
            st.error(f"Lỗi khi tạo câu trả lời: {e}")
