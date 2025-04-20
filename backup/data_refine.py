import os
import re
import logging
from pathlib import Path
from docx import Document
from pdfminer.high_level import extract_text

# 🔕 Tắt warning từ pdfminer (vd: CropBox missing...)
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# 📁 Đường dẫn tương đối trong thư mục `data/`
RAW_DIR = Path("raw")
REFINED_DIR = Path("refined")

SUPPORTED_EXTS = [".pdf", ".docx", ".txt"]

refined_count = 0
total_files = 0
written_dirs = set()

def clean_text(text):
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def split_sentences(text):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    paragraphs = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) < 512:
            current += sent + " "
        else:
            paragraphs.append(current.strip())
            current = sent + " "
    if current:
        paragraphs.append(current.strip())
    return "\n\n".join(paragraphs)

def extract_text_from_file(file_path: Path) -> str:
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        return extract_text(str(file_path))
    elif ext == ".docx":
        doc = Document(str(file_path))
        return "\n".join([p.text for p in doc.paragraphs])
    elif ext == ".txt":
        return Path(file_path).read_text(encoding="utf-8", errors="ignore")
    else:
        return ""

def process_file(file_path: Path):
    global refined_count
    rel_path = file_path.relative_to(RAW_DIR)
    output_path = REFINED_DIR / rel_path.with_suffix(".txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        raw_text = extract_text_from_file(file_path)
        cleaned = clean_text(raw_text)
        segmented = split_sentences(cleaned)
        output_path.write_text(segmented, encoding="utf-8")
        print(f"✅ Refined: {file_path} → {output_path}")
        refined_count += 1
        written_dirs.add(str(output_path.parent))
    except Exception as e:
        print(f"❌ Lỗi xử lý {file_path}: {e}")

def scan_and_process():
    global total_files
    files = list(RAW_DIR.rglob("*"))
    for file_path in files:
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTS:
            total_files += 1
            process_file(file_path)

    print("\n📊 Tổng kết:")
    print(f"🔍 Số file tìm thấy: {total_files}")
    print(f"✍️  Số file đã refine: {refined_count}")
    print("📂 Ghi vào các thư mục:")
    for d in sorted(written_dirs):
        print(f"   - {d}")

if __name__ == "__main__":
    scan_and_process()
