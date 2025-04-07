# sample_test_onnx.py
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("face_security.onnx")
dummy = np.random.randn(1,3,160,160).astype(np.float32)
outputs = sess.run(None, {"input": dummy})
print("Output shape:", outputs[0].shape)  # nên ra (1,512)
