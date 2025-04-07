import torch
import onnx
from facenet_pytorch import InceptionResnetV1

"""
Script export model InceptionResnetV1 (FaceNet) sang ONNX để test local.
Đặt file này cùng thư mục với face_recognition_train.py.
"""

def export_to_onnx(onnx_filename="face_security.onnx"):
    # 1. Load pretrained FaceNet
    model = InceptionResnetV1(pretrained='vggface2').eval()
    # 2. Tạo dummy input (batch=1, 3 channel, 160x160)
    dummy_input = torch.randn(1, 3, 160, 160)
    # 3. Export sang ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_filename,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        do_constant_folding=True
    )
    print(f"Exported ONNX model to {onnx_filename}")

    # 4. Kiểm tra model ONNX
    onnx_model = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid.")

if __name__ == '__main__':
    export_to_onnx()
