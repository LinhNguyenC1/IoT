"""
Công Cụ Đăng Ký Khuôn Mặt Mẫu (Face Enrollment Tool)
Hỗ trợ:
1. Chụp trực tiếp từ Webcam USB (lấy 5 mẫu góc mặt, tính vector trung bình tối ưu độ chính xác)
2. Quét nạp hàng loạt từ thư mục ảnh faces/<Ten_Nguoi>/
3. Xem danh sách hoặc xóa người dùng khỏi SQLite
"""

import os
import sys
import time
import argparse
from pathlib import Path
import cv2
import numpy as np

from database.local_db import LocalDB
from detection import create_detector
from alignment import align_face
from recognition import FaceEmbedder


def enroll_from_webcam(name: str, camera_index: int = 0, num_samples: int = 5, detector_type: str = None):
    if detector_type is None:
        detector_type = os.getenv("APP_DETECTOR", "scrfd").strip().lower()
    has_display = bool(os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY"))

    print("\n" + "=" * 60)
    print(f" ĐĂNG KÝ KHUÔN MẶT QUA WEBCAM: {name}")
    print("=" * 60)
    if has_display:
        print(" Hướng dẫn:")
        print(" - Nhìn thẳng vào camera, giữ khuôn mặt ở giữa khung hình.")
        print(" - Hơi nghiêng nhẹ góc mặt khi hệ thống chụp các mẫu.")
        print(" - Nhấn 's' để bắt đầu chụp mẫu | Nhấn 'q' để hủy.")
    else:
        print(" [CHẾ ĐỘ HEADLESS / SSH] Không phát hiện màn hình đồ họa ($DISPLAY).")
        print(" - Hãy ngồi đối diện trước USB Webcam và nhìn thẳng.")
        try:
            input(" -> Nhấn [ENTER] trên bàn phím để bắt đầu tự động chụp mẫu... ")
        except (KeyboardInterrupt, EOFError):
            print("\nĐã hủy.")
            return False
    print("=" * 60 + "\n")

    db = LocalDB()
    detector = create_detector(detector_type=detector_type, input_size=(640, 480))
    embedder = FaceEmbedder()

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print(f"[LỖI] Không thể mở webcam tại /dev/video{camera_index}!")
        return False

    collected_embeddings = []
    capturing = not has_display  # Nếu headless thì tự động chụp ngay
    last_sample_time = 0
    start_time = time.time()
    last_wait_msg_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            h, w = frame.shape[:2]

            # Dò khuôn mặt
            faces = detector.detect(frame)

            # Chọn khuôn mặt lớn nhất
            main_face = None
            if len(faces) > 0:
                faces.sort(key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]), reverse=True)
                main_face = faces[0]

            now = time.time()

            if main_face:
                landmarks = main_face["landmarks"]
                if capturing:
                    if now - last_sample_time >= 0.35:
                        aligned = align_face(frame, landmarks)
                        if aligned is not None:
                            feat = embedder.extract(aligned)
                            if feat is not None:
                                collected_embeddings.append(feat)
                                last_sample_time = now
                                print(f"  [+] Đã chụp mẫu {len(collected_embeddings)}/{num_samples}...")

                        if len(collected_embeddings) >= num_samples:
                            break
            else:
                if not has_display and (now - last_wait_msg_time >= 1.5):
                    print("  [Đang chờ] Vui lòng nhìn thẳng vào camera...")
                    last_wait_msg_time = now

            # Nếu có màn hình đồ họa -> Hiển thị OpenCV GUI
            if has_display:
                display_frame = frame.copy()
                if main_face:
                    x1, y1, x2, y2 = main_face["bbox"]
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    for pt in main_face["landmarks"]:
                        cv2.circle(display_frame, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)

                status_text = (
                    f"Dang ky: {name} | [S]: Chup mau | [Q]: Huy"
                    if not capturing
                    else f"Dang thu thap: {len(collected_embeddings)}/{num_samples}..."
                )
                cv2.rectangle(display_frame, (0, 0), (w, 40), (35, 35, 35), -1)
                cv2.putText(display_frame, status_text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA)

                try:
                    cv2.imshow("SmartFace - Dang ky khuon mat", display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        print("[HỦY] Người dùng đã hủy đăng ký.")
                        return False
                    elif key == ord("s") and not capturing:
                        if main_face:
                            capturing = True
                            print("[BẮT ĐẦU] Đang tự động chụp mẫu...")
                        else:
                            print("[CẢNH BÁO] Chưa tìm thấy khuôn mặt trong camera để bắt đầu!")
                except Exception:
                    has_display = False  # Nếu lỗi display giữa chừng thì fallback về headless
            else:
                # Chế độ headless timeout an toàn sau 30s nếu không thấy mặt
                if now - start_time > 30.0 and len(collected_embeddings) == 0:
                    print("[TIMEOUT] Quá 30 giây không phát hiện khuôn mặt trước webcam!")
                    break
                time.sleep(0.01)

    finally:
        cap.release()
        if has_display:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    if len(collected_embeddings) >= num_samples:
        # Tính vector trung bình của các mẫu
        avg_vec = np.mean(collected_embeddings, axis=0)
        norm = np.linalg.norm(avg_vec)
        if norm > 1e-6:
            avg_vec = avg_vec / norm

        db.add_user(name, avg_vec)
        print("\n" + "=" * 60)
        print(f" [THÀNH CÔNG] Đã lưu {num_samples} vector khuôn mặt cho: {name}")
        print(" Cơ sở dữ liệu SQLite: faces.db đã được cập nhật!")
        print("=" * 60 + "\n")
        return True
    else:
        print("[LỖI] Không thu thập đủ số lượng mẫu hợp lệ!")
        return False


def enroll_from_folder(folder_path: str = "faces", detector_type: str = None):
    if detector_type is None:
        detector_type = os.getenv("APP_DETECTOR", "scrfd").strip().lower()
    path = Path(folder_path)
    if not path.is_absolute():
        script_dir = Path(__file__).resolve().parent
        if (script_dir / folder_path).exists():
            path = script_dir / folder_path

    if not path.exists() or not path.is_dir():
        print(f"[LỖI] Thư mục không tồn tại: {folder_path} (Đường dẫn: {path})")
        return False

    print("\n" + "=" * 60)
    print(f" QUÉT NẠP KHUÔN MẶT TỪ THƯ MỤC: {folder_path}")
    print("=" * 60)

    db = LocalDB()
    detector = create_detector(detector_type=detector_type, input_size=(640, 640))
    embedder = FaceEmbedder()

    # Tìm các thư mục con hoặc file ảnh
    enrolled_count = 0
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for person_dir in sorted(path.iterdir()):
        if person_dir.is_dir():
            name = person_dir.name
            image_files = [f for f in person_dir.iterdir() if f.suffix.lower() in valid_exts]
            if not image_files:
                continue

            person_embeddings = []
            for img_file in image_files:
                img = cv2.imread(str(img_file))
                if img is None:
                    continue

                faces = detector.detect(img)
                if faces:
                    faces.sort(key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]), reverse=True)
                    aligned = align_face(img, faces[0]["landmarks"])
                    if aligned is not None:
                        feat = embedder.extract(aligned)
                        if feat is not None:
                            person_embeddings.append(feat)

            if person_embeddings:
                avg_vec = np.mean(person_embeddings, axis=0)
                norm = np.linalg.norm(avg_vec)
                if norm > 1e-6:
                    avg_vec = avg_vec / norm
                db.add_user(name, avg_vec)
                enrolled_count += 1
                print(f"  [OK] Đã nạp thành công '{name}' ({len(person_embeddings)} ảnh)")

    print(f"\n[HOÀN TẤT] Tổng cộng đã nạp {enrolled_count} người vào database faces.db!\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Công cụ đăng ký khuôn mặt cho Raspberry Pi 4")
    parser.add_argument("--name", type=str, help="Họ và tên người cần đăng ký")
    parser.add_argument(
        "--folder",
        nargs="?",
        const="faces",
        type=str,
        help="Đường dẫn thư mục ảnh để nạp hàng loạt (mặc định: 'faces')",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index (mặc định 0)")
    parser.add_argument(
        "--detector",
        type=str,
        default=os.getenv("APP_DETECTOR", "scrfd"),
        help="Model detector: scrfd hoặc yunnet (mặc định: scrfd)",
    )
    parser.add_argument("--samples", type=int, default=5, help="Số lượng mẫu chụp (mặc định 5)")
    parser.add_argument("--list", action="store_true", help="Xem danh sách người dùng đã đăng ký")
    parser.add_argument("--delete", type=str, help="Xóa một người dùng theo tên")

    args = parser.parse_args()
    db = LocalDB()

    # Xem danh sách
    if args.list:
        users = db.list_users()
        print("\n" + "=" * 50)
        print(" DANH SÁCH NGƯỜI DÙNG TRONG DATABASE (faces.db)")
        print("=" * 50)
        if not users:
            print(" (Chưa có người dùng nào được đăng ký)")
        else:
            for u in users:
                print(f" ID: {u['id']:<3} | Tên: {u['name']:<25} | Mẫu: {u['samples']} | Ngày tạo: {u['created_at']}")
        print("=" * 50 + "\n")
        return

    # Xóa người dùng
    if args.delete:
        name = args.delete.strip()
        confirm = input(f"Bạn có chắc muốn xóa người dùng '{name}' không? (y/N): ").strip().lower()
        if confirm == "y":
            deleted = db.delete_user(name)
            if deleted:
                print(f"[OK] Đã xóa thành công '{name}'.")
            else:
                print(f"[LỖI] Không tìm thấy người dùng '{name}'!")
        return

    # Nạp từ thư mục
    if args.folder:
        enroll_from_folder(args.folder, detector_type=args.detector)
        return

    # Đăng ký qua Webcam
    name = args.name
    if not name:
        try:
            name = input("Nhập Họ và Tên người cần đăng ký: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nĐã hủy.")
            return

    if not name:
        print("[LỖI] Tên người dùng không được để trống!")
        return

    enroll_from_webcam(name, camera_index=args.camera, num_samples=args.samples, detector_type=args.detector)


if __name__ == "__main__":
    main()
