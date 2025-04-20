import os
import sys
import time
import json
import faiss
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

REFINED_DIR = Path("refined")
INDEX_DIR = Path("index")
MODEL_DIR = Path("encoder/bge-m3")
BATCH_SIZE = 32

def load_model():
    import torch
    if not MODEL_DIR.exists() or not (MODEL_DIR / "config.json").exists():
        print("❌ Không tìm thấy mô hình tại: encoder/bge-m3")
        exit(1)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"✅ Dùng {'GPU Mac (MPS)' if device == 'mps' else 'CPU'}")
    try:
        model = SentenceTransformer(str(MODEL_DIR), device=device)
        print("✅ Đã load model thành công từ: encoder/bge-m3")
        return model
    except Exception as e:
        print("❌ Lỗi khi load mô hình embedding:")
        print(e)
        exit(1)

def get_chunks(text):
    return [c.strip() for c in text.split("\n\n") if len(c.strip()) > 30]

def encode_chunks(chunks, model):
    sys.stdout.write(f"🧠 Đang encode {len(chunks)} đoạn...\033[K\r")
    sys.stdout.flush()
    embeddings = model.encode(
        [f"Represent this sentence for searching relevant passages: {c}" for c in chunks],
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=False
    )
    return np.array(embeddings)

def index_folder(folder, model, bar):
    all_texts, all_embeds = [], []
    txt_files = list(folder.glob("*.txt"))
    if not txt_files:
        return 0

    for txt_file in txt_files:
        text = txt_file.read_text(encoding="utf-8", errors="ignore")
        chunks = get_chunks(text)
        if not chunks:
            bar.update(1)
            continue

        # ✅ Đặt log dưới thanh progress bar, ghi đè mỗi lần
        bar.write(f"📄 File: {txt_file.relative_to(REFINED_DIR)}")
        bar.write(f"🧠 Đang encode {len(chunks)} đoạn...\033[K\r")

        embeddings = encode_chunks(chunks, model)
        all_texts.extend(chunks)
        all_embeds.append(embeddings)
        bar.update(1)

    if not all_embeds:
        return 0

    vectors = np.vstack(all_embeds)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    out_dir = INDEX_DIR / folder.relative_to(REFINED_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "index.faiss"))
    with open(out_dir / "mapping.json", "w", encoding="utf-8") as f:
        json.dump(all_texts, f, ensure_ascii=False, indent=2)

    bar.write(f"✅ Đã index {len(all_texts)} đoạn → {out_dir}/index.faiss")
    return len(txt_files)

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    start_all = time.time()

    folders = sorted({f.parent for f in REFINED_DIR.rglob("*.txt")})
    files = list(REFINED_DIR.rglob("*.txt"))
    print(f"🔍 Tìm thấy {len(folders)} thư mục | {len(files)} file cần index")

    model = load_model()
    total_indexed = 0

    with tqdm(total=len(files), desc="📦 Indexing toàn bộ", ncols=80) as bar:
        for folder in folders:
            count = index_folder(folder, model, bar)
            total_indexed += count
        bar.close()
        print()

    print(f"✅ Hoàn tất. Đã index {total_indexed} file.")
    print(f"📂 FAISS index lưu tại: {INDEX_DIR}/")
    print(f"🕓 Tổng thời gian: {time.time() - start_all:.2f} giây")

if __name__ == "__main__":
    main()
