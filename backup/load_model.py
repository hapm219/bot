import os, sys, time, gc, requests, contextlib
from tqdm import tqdm
from llama_cpp import Llama

MODEL_URL = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MODEL_PATH = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

def download_model():
    os.makedirs("models", exist_ok=True)
    if os.path.exists(MODEL_PATH): return
    print(f"⬇️ Đang tải mô hình từ: {MODEL_URL}")
    with requests.get(MODEL_URL, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with open(MODEL_PATH, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="📦 Tải về") as bar:
            for chunk in r.iter_content(8192):
                f.write(chunk)
                bar.update(len(chunk))
    print(f"✅ Đã tải xong: {MODEL_PATH}")

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as fnull:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = fnull
        try: yield
        finally: sys.stdout, sys.stderr = old_out, old_err

def load_llm():
    download_model()
    print(f"\n🧠 Sẵn sàng tạo mô hình llama.cpp từ: {MODEL_PATH}")

    def safe_infer(prompt, max_tokens=256, **kwargs):
        with suppress_output():
            llm = Llama(
                model_path=MODEL_PATH,
                n_ctx=1024,
                n_threads=os.cpu_count(),
                n_gpu_layers=32,
                use_mlock=True,
                verbose=False
            )
        result = llm(prompt, max_tokens=max_tokens, **kwargs)
        del llm; gc.collect()
        return result

    return safe_infer
