from optimum.exporters.onnx import main_export

main_export(
    model_name_or_path="BAAI/bge-m3",
    output="onnx/bge-m3-fixed",
    task="feature-extraction",
    use_external_data_format=False  # ❗ Bắt buộc để không sinh ra .onnx_data
)
