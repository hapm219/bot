from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bge-m3", use_fast=True)
tokenizer.save_pretrained("onnx/bge-m3-fixed")
print("✅ Tokenizer đã ghi đè thành công vào bge-m3-fixed")
