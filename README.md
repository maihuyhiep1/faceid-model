Cài thư viện 
pip install torch torchvision facenet-pytorch pillow numpy joblib

Enroll người dùng

python face_recognition_train.py enroll alice --camera

Verify người dùng 

python face_recognition_train.py verify alice --camera

Xuất lại model ONNX (nếu muốn dùng model mới)

python convert_to_onnx.py

