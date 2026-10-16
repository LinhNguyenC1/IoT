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


def enroll_from_webcam(name: str, camera_index: int = 0, num_samples: int = 5):
    has_display = bool(os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY"))

    print("\n" + "=" * 60)
    print(f" ĐĂNG KÝ KHUÔN MẶT QUA WEBCAM: {name}")
    print("=" * 60)
    if has_display:
        print(" Hướng dẫn:")
        print(f" - Nhìn vào camera và bấm phím 's' trên cửa sổ để chụp từng ảnh (cần đủ {num_samples} ảnh).")
        print(" - Giữa các lần bấm 's', bạn có thể hơi đổi góc mặt (thẳng, nghiêng nhẹ) để lấy mẫu đa dạng.")
        print(" - Nhấn 'q' để hủy bỏ đăng ký.")
    else:
        print(" [CHẾ ĐỘ HEADLESS / SSH] Không phát hiện màn hình đồ họa ($DISPLAY).")
        print(f" - Nhấn [ENTER] trên bàn phím mỗi lần muốn chụp 1 ảnh mẫu (cần đủ {num_samples} ảnh).")
    print("=" * 60 + "\n")

    db = LocalDB()
    detector = create_detector(det_size=(640, 640))
    embedder = FaceEmbedder()

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print(f"[LỖI] Không thể mở webcam tại /dev/video{camera_index}!")
        return False

    collected_embeddings = []
    collected_frames = []

    try:
        while len(collected_embeddings) < num_samples:
            if has_display:
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                h, w = frame.shape[:2]

                # Dò khuôn mặt thời gian thực
                faces = detector.detect(frame)
                main_face = None
                if len(faces) > 0:
                    faces.sort(key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]), reverse=True)
                    main_face = faces[0]

                display_frame = frame.copy()
                if main_face:
                    x1, y1, x2, y2 = main_face["bbox"]
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    for pt in main_face["landmarks"]:
                        cv2.circle(display_frame, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)

                header_h = 42
                cv2.rectangle(display_frame, (0, 0), (w, header_h), (35, 35, 35), -1)
                status_text = f"Dang ky: {name} | Da chup: {len(collected_embeddings)}/{num_samples} | Nhan [S] de chup | [Q] Huy"
                cv2.putText(display_frame, status_text, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)

                try:
                    cv2.imshow("SmartFace - Dang ky khuon mat", display_frame)
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("q") or key == 27:
                        print("[HỦY] Người dùng đã hủy đăng ký.")
                        return False
                    elif key in (ord("s"), ord("S")):
                        if main_face:
                            aligned = align_face(frame, main_face["landmarks"])
                            if aligned is not None:
                                feat = embedder.extract(aligned)
                                if feat is not None:
                                    collected_embeddings.append(feat)
                                    collected_frames.append(frame.copy())
                                    print(f"  [+] Đã chụp mẫu {len(collected_embeddings)}/{num_samples} thành công!")
                                else:
                                    print("  [CẢNH BÁO] Không trích xuất được đặc trưng, vui lòng nhấn 's' lại.")
                            else:
                                print("  [CẢNH BÁO] Căn chỉnh khuôn mặt thất bại, vui lòng nhấn 's' lại.")

                            if len(collected_embeddings) >= num_samples:
                                break
                        else:
                            print("  [CẢNH BÁO] Không tìm thấy khuôn mặt trong camera! Vui lòng nhìn thẳng và nhấn 's'.")
                except Exception:
                    has_display = False
            else:
                # Chế độ Headless / SSH: Chụp từng ảnh khi người dùng bấm ENTER
                try:
                    prompt_str = f"Nhấn [ENTER] để chụp ảnh mẫu {len(collected_embeddings) + 1}/{num_samples} (hoặc gõ 'q' để hủy): "
                    user_cmd = input(prompt_str).strip().lower()
                    if user_cmd == "q":
                        print("[HỦY] Người dùng đã hủy đăng ký.")
                        return False

                    # Đọc frame mới nhất từ webcam
                    for _ in range(5):
                        cap.read()
                    ret, cur_frame = cap.read()
                    if not ret or cur_frame is None:
                        print("  [LỖI] Không thể đọc hình ảnh từ webcam!")
                        continue

                    faces = detector.detect(cur_frame)
                    if not faces:
                        print("  [CẢNH BÁO] Không phát hiện khuôn mặt trước webcam! Vui lòng thử lại.")
                        continue

                    faces.sort(key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]), reverse=True)
                    aligned = align_face(cur_frame, faces[0]["landmarks"])
                    if aligned is not None:
                        feat = embedder.extract(aligned)
                        if feat is not None:
                            collected_embeddings.append(feat)
                            collected_frames.append(cur_frame.copy())
                            print(f"  [+] Đã chụp mẫu {len(collected_embeddings)}/{num_samples} thành công!")
                        else:
                            print("  [CẢNH BÁO] Không trích xuất được đặc trưng, vui lòng thử lại.")
                    else:
                        print("  [CẢNH BÁO] Căn chỉnh khuôn mặt thất bại, vui lòng thử lại.")
                except (KeyboardInterrupt, EOFError):
                    print("\n[HỦY] Đã dừng đăng ký.")
                    return False

    finally:
        cap.release()
        if has_display:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    if len(collected_embeddings) >= num_samples:
        # 1. Tính vector trung bình của các mẫu
        avg_vec = np.mean(collected_embeddings, axis=0)
        norm = np.linalg.norm(avg_vec)
        if norm > 1e-6:
            avg_vec = avg_vec / norm

        db.add_user(name, avg_vec)

        # 2. Lưu ảnh chụp từ webcam vào thư mục faces/<name>/
        faces_dir = Path(__file__).resolve().parent / "faces"
        safe_name = "".join(c for c in name if c not in r'\/:*?"<>|').strip()
        user_faces_dir = faces_dir / (safe_name if safe_name else "user")
        user_faces_dir.mkdir(parents=True, exist_ok=True)

        for idx, img in enumerate(collected_frames, start=1):
            img_path = user_faces_dir / f"{idx}.jpg"
            cv2.imwrite(str(img_path), img)

        print("\n" + "=" * 60)
        print(f" [THÀNH CÔNG] Đã đăng ký thành công cho: {name}")
        print(f" - Cơ sở dữ liệu: Đã lưu vector 512D vào faces.db")
        print(f" - Lưu trữ ảnh  : Đã lưu {len(collected_frames)} ảnh vào thư mục '{user_faces_dir}'")
        print("=" * 60 + "\n")
        return True
    else:
        print("[LỖI] Không thu thập đủ số lượng mẫu hợp lệ!")
        return False


def enroll_from_folder(folder_path: str = "faces"):
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
    detector = create_detector(det_size=(640, 640))
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
                print(f"[OK] Đã xóa thành công '{name}' trong database faces.db.")
                faces_user_dir = Path(__file__).resolve().parent / "faces" / name
                if faces_user_dir.exists() and faces_user_dir.is_dir():
                    import shutil
                    shutil.rmtree(faces_user_dir, ignore_errors=True)
                    print(f"[OK] Đã xóa thư mục ảnh '{faces_user_dir}'.")
            else:
                print(f"[LỖI] Không tìm thấy người dùng '{name}'!")
        return

    # Nạp từ thư mục
    if args.folder:
        enroll_from_folder(args.folder)
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

    enroll_from_webcam(name, camera_index=args.camera, num_samples=args.samples)


if __name__ == "__main__":
    main()
