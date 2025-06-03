# export.py
import torch
import torch_tensorrt
from models import SignTransformer


def export_onnx(model, seq_len=64, path="model.onnx"):
    dummy = torch.randn(1, seq_len, 1662).to(next(model.parameters()).device)
    torch.onnx.export(model, dummy, path, export_params=True, opset_version=13)
    print("ONNX export completed.")


def optimize_tensorrt(onnx_path, trt_path="model_trt.ts"):
    engine = torch_tensorrt.compile(onnx_path, inputs=[torch.randn(1,64,33*4).cuda()])
    torch.save(engine, trt_path)
    print("TensorRT optimization completed.")


# main.py
from pathlib import Path
from train import cross_validate

def main():
    data_dir = Path('data')
    all_files = list(data_dir.rglob('*.mp4'))
    labels = [p.parent.name for p in all_files]
    uniq = sorted(set(labels))
    label_map = {l:i for i,l in enumerate(uniq)}
    y = [label_map[l] for l in labels]
    cross_validate(all_files, y, label_map, seq_len=64)

if __name__ == '__main__':
    main()
