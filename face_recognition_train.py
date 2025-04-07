import os
import argparse
from PIL import Image
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import joblib
import cv2

# Khởi tạo MTCNN và FaceNet (InceptionResnetV1)
mtcnn = MTCNN(image_size=160, margin=0)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def capture_frame(camera_index=0):
    """
    Mở camera, hiển thị và chụp frame khi nhấn 'q'.
    Trả về PIL Image.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Không thể mở camera")
    print("Nhấn 'q' để chụp ảnh...")
    frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    # Chuyển BGR->RGB và sang PIL
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def get_embedding(img_input):
    """
    img_input: đường dẫn file hoặc PIL Image
    Trả về embedding 512-d.
    """
    if isinstance(img_input, str):
        img = Image.open(img_input).convert('RGB')
    else:
        img = img_input.convert('RGB')
    face = mtcnn(img)
    if face is None:
        raise ValueError("Không phát hiện khuôn mặt")
    with torch.no_grad():
        emb = resnet(face.unsqueeze(0)).cpu().numpy().flatten()
    return emb


def enroll(user_id, img_path, use_camera, db_path):
    """
    Đăng ký FaceID cho user_id: nếu use_camera=True thì chụp từ webcam,
    ngược lại dùng file ảnh.
    Lưu template embedding.
    """
    try:
        db = joblib.load(db_path)
    except:
        db = {}

    if use_camera:
        img = capture_frame()
        emb = get_embedding(img)
    else:
        emb = get_embedding(img_path)

    db[user_id] = emb
    joblib.dump(db, db_path)
    print(f"Đã enroll '{user_id}' vào {db_path}")


def verify(user_id, img_path, use_camera, db_path, threshold):
    """
    Xác thực khuôn mặt: chụp hoặc dùng file ảnh, so sánh với template.
    """
    if not os.path.exists(db_path):
        print("Chưa có database enroll.")
        return False
    db = joblib.load(db_path)
    if user_id not in db:
        print(f"User '{user_id}' chưa enroll.")
        return False

    if use_camera:
        img = capture_frame()
        emb = get_embedding(img)
    else:
        emb = get_embedding(img_path)

    dist = np.linalg.norm(emb - db[user_id])
    print(f"Khoảng cách: {dist:.4f}")
    if dist < threshold:
        print("Xác thực thành công.")
        return True
    else:
        print("Xác thực thất bại.")
        return False


def main():
    parser = argparse.ArgumentParser(description="FaceID Enrollment & Verification")
    sub = parser.add_subparsers(dest="mode", required=True)

    # Enroll
    p1 = sub.add_parser("enroll", help="Enroll user with image or camera")
    p1.add_argument("user_id", help="User identifier")
    p1.add_argument("--image", help="Path to image file", default=None)
    p1.add_argument("--camera", action="store_true", help="Capture from webcam")
    p1.add_argument("--db", default="templates.joblib", help="Path to templates DB")

    # Verify
    p2 = sub.add_parser("verify", help="Verify user with image or camera")
    p2.add_argument("user_id", help="User identifier")
    p2.add_argument("--image", help="Path to image file", default=None)
    p2.add_argument("--camera", action="store_true", help="Capture from webcam")
    p2.add_argument("--db", default="templates.joblib", help="Path to templates DB")
    p2.add_argument("--threshold", type=float, default=0.6, help="Distance threshold")

    args = parser.parse_args()
    if args.mode == "enroll":
        if not args.image and not args.camera:
            print("Cần --image hoặc --camera để enroll.")
            return
        enroll(args.user_id, args.image, args.camera, args.db)
    else:
        if not args.image and not args.camera:
            print("Cần --image hoặc --camera để verify.")
            return
        verify(args.user_id, args.image, args.camera, args.db, args.threshold)

if __name__ == "__main__":
    main()
