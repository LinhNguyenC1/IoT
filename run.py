"""
Chương Trình Nhận Diện Khuôn Mặt & Chống Giả Mạo Tầng Biên (Pi 4 Edge Face Recognition)
Tích Hợp Hệ Thống Kiểm Soát Cửa Tự Động:
- Cảm biến siêu âm HC-SR04: Có vật cản phía trước mới kích hoạt nhận diện khuôn mặt
- Nhận diện đúng (Người thật + Có đăng ký trong CSDL): Quay Servo 90 độ mở cửa
- Phát hiện giả mạo (Ảnh in, màn hình điện thoại): Bật đèn ĐỎ, giữ Servo 0 độ
- Người thật nhưng không có trong CSDL (Unknown): Bật đèn XANH, giữ Servo 0 độ
"""

import os
import sys
import time
import signal
import argparse
from pathlib import Path
import cv2
import numpy as np
from dotenv import load_dotenv

# Tải cấu hình .env
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv()

from database.local_db import LocalDB
from detection import create_detector
from tracking import FaceTracker
from alignment import align_face
from recognition import FaceEmbedder
from antispoof import AntiSpoofPredictor
from peripherals.access_controller import AccessController


def parse_args():
    parser = argparse.ArgumentParser(description="Pi 4 Edge Face Recognition & Smart Access Control")
    parser.add_argument("--pad", dest="enable_pad", action="store_true", help="Bật chế độ Chống giả mạo (PAD)")
    parser.add_argument("--no-pad", dest="enable_pad", action="store_false", help="Tắt chế độ Chống giả mạo (Không PAD)")
    parser.set_defaults(enable_pad=None)

    parser.add_argument("--camera", type=int, default=int(os.getenv("CAMERA_INDEX", "0")), help="Camera index")
    parser.add_argument("--detector", type=str, default="scrfd", help="Bộ dò khuôn mặt (mặc định: scrfd)")
    parser.add_argument("--threshold", type=float, default=float(os.getenv("MATCH_THRESHOLD", "0.40")), help="Ngưỡng so khớp ArcFace")
    parser.add_argument("--headless", action="store_true", help="Chạy không giao diện đồ họa (in log terminal)")
    parser.add_argument("--distance-threshold", type=float, default=float(os.getenv("OBSTACLE_DISTANCE_CM", "40.0")), help="Khoảng cách phát hiện vật cản (cm)")
    parser.add_argument("--door-hold", type=float, default=float(os.getenv("DOOR_HOLD_SECONDS", "3.0")), help="Thời gian giữ mở cửa (giây)")
    parser.add_argument("--simulate-obstacle", action="store_true", help="Bật mô phỏng có vật cản (dùng khi test không có cảm biến siêu âm)")
    return parser.parse_args()


class EdgeApp:
    def __init__(self, args):
        self.args = args

        # 1. Trạng thái PAD (Mặc định bật để chống giả mạo)
        if args.enable_pad is not None:
            self.enable_pad = args.enable_pad
        else:
            self.enable_pad = os.getenv("ENABLE_PAD", "true").strip().lower() in ("true", "1", "yes")

        # Tự động phát hiện môi trường đồ họa ($DISPLAY)
        has_display = bool(os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY"))
        if not self.args.headless and not has_display:
            print("[HEADLESS TỰ ĐỘNG] Không phát hiện biến $DISPLAY (chạy qua SSH). Tự động chuyển sang chế độ Console Log.")
            self.args.headless = True

        # 2. Khởi tạo Database SQLite
        self.db = LocalDB()
        user_count = len(self.db.list_users())
        print(f"[Database] Đã tải {user_count} người dùng từ SQLite (faces.db).")

        # 3. Khởi tạo AI Modules
        self.detector_type = "scrfd"
        self.detector = create_detector(det_size=(640, 640), conf_thresh=0.5)
        self.tracker = FaceTracker(
            recognize_interval_seconds=1.0,
            iou_threshold=0.3,
            max_missing_frames=10,
            pad_smooth_window=5,
            pad_spoof_min_ratio=0.6,
            pad_interval_seconds=0.2,
            pad_min_votes=3,
            pad_stale_timeout=3.0,
        )
        self.embedder = FaceEmbedder()
        self.antispoof = AntiSpoofPredictor(threshold_logit=float(os.getenv("FAS_LOGIT_THRESHOLD", "-0.3730")))
        print(f"[AI Modules] Nhận diện : ArcFace | Ngưỡng MATCH_THRESHOLD = {self.args.threshold}")
        print(f"[AI Modules] Anti-Spoof: {self.antispoof.model_name} | Ngưỡng FAS_LOGIT_THRESHOLD = {self.antispoof.threshold_logit}")

        # 4. Khởi tạo Bộ Điều Khiển Cửa & Ngoại Vi (Siêu âm + Servo + 2 LED)
        self.access_ctrl = AccessController(
            obstacle_distance_cm=self.args.distance_threshold,
            door_hold_seconds=self.args.door_hold,
        )
        if self.args.simulate_obstacle:
            self.access_ctrl.simulated_obstacle = True
            print("[AccessController] Đã bật chế độ MÔ PHỎNG VẬT CẢN (Simulated Obstacle = True).")

        # 5. Cấu hình Camera
        self.cap = cv2.VideoCapture(args.camera)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not self.cap.isOpened():
            print(f"[LỖI] Không thể kết nối với Webcam tại /dev/video{args.camera}!")
            sys.exit(1)

        self.track_identities = {}  # track_id -> (name, score)
        self.running = True
        self.detection_interval = 0.2
        self.last_detection_time = 0.0

    def toggle_pad_mode(self):
        """Đổi chế độ qua lại giữa CÓ PAD và KHÔNG PAD."""
        self.enable_pad = not self.enable_pad
        self.track_identities.clear()
        self.access_ctrl.reset_all()

        mode_name = "CÓ PAD (BẢO MẬT CHỐNG GIẢ MẠO)" if self.enable_pad else "KHÔNG PAD (TỐC ĐỘ CAO)"
        print(f"\n[CHUYỂN CHẾ ĐỘ] -> {mode_name}\n")

    def toggle_simulate_obstacle(self):
        """Chuyển đổi trạng thái mô phỏng vật cản khi test trên máy dev."""
        self.access_ctrl.simulated_obstacle = not self.access_ctrl.simulated_obstacle
        st = "CÓ VẬT CẢN" if self.access_ctrl.simulated_obstacle else "KHÔNG CÓ VẬT CẢN"
        print(f"\n[MÔ PHỎNG SIÊU ÂM] Đã chuyển đổi -> {st}\n")

    def run(self):
        print("\n" + "=" * 70)
        print(" HỆ THỐNG KIỂM SOÁT CỬA TỰ ĐỘNG BẰNG AI & CẢM BIẾN (PI 4)")
        print("=" * 70)
        print(f" - Điều kiện kích hoạt: Cảm biến siêu âm phát hiện vật cản <= {self.access_ctrl.obstacle_distance_cm} cm")
        print(f" - Nhận diện ĐÚNG     : Quay Servo 90 độ mở cửa (giữ {self.access_ctrl.door_hold_seconds}s)")
        print(f" - Phát hiện GIẢ MẠO  : Bật Đèn ĐỎ (Đóng servo 0 độ)")
        print(f" - Người lạ (UNKNOWN) : Bật Đèn XANH (Đóng servo 0 độ)")
        print(" Phím tắt:")
        print("   [p] : Bật/Tắt chế độ kiểm tra chống giả mạo (PAD)")
        print("   [o] : Bật/Tắt mô phỏng vật cản siêu âm (Test mô phỏng)")
        print("   [q] : Thoát chương trình")
        print("=" * 70 + "\n")

        fps = 0.0
        frame_count = 0
        fps_timer = time.time()
        last_log_time = 0.0

        try:
            while self.running:
                # ── 0. Cập nhật chu kỳ ngoại vi (Tự động đóng cửa sau timeout) ──
                self.access_ctrl.update()

                ret, frame = self.cap.read()
                if not ret or frame is None:
                    continue

                now_time = time.time()
                frame_count += 1
                if now_time - fps_timer >= 1.0:
                    fps = frame_count / (now_time - fps_timer)
                    frame_count = 0
                    fps_timer = now_time

                # ── 1. Đọc Cảm Biến Siêu Âm: Kiểm tra Vật Cản ──
                distance_cm = self.access_ctrl.get_distance_cm()
                has_obstacle = self.access_ctrl.has_obstacle()

                tracked = []
                system_status_msg = "CHỜ VẬT CẢN..."
                system_status_color = (180, 180, 180)

                # NẾU CÓ VẬT CẢN HOẶC CỬA ĐANG MỞ: KÍCH HOẠT NHẬN DIỆN
                if has_obstacle or self.access_ctrl.is_door_open:
                    system_status_msg = "PHÁT HIỆN VẬT CẢN - ĐANG QUÉT MẶT..."
                    system_status_color = (0, 215, 255)

                    # ── 2. Dò mặt cách quãng + Lucas-Kanade Optical Flow Tracker ──
                    need_detect = (now_time - self.last_detection_time) >= self.detection_interval
                    if need_detect or self.tracker.needs_redetection():
                        detections = self.detector.detect(frame)
                        self.last_detection_time = now_time
                        tracked = self.tracker.update(detections, frame=frame, detector_called=True)
                    else:
                        tracked = self.tracker.update([], frame=frame, detector_called=False)

                    active_track_ids = {t.track_id for _, t in tracked}

                    # Dọn dẹp các track đã rời đi
                    for tid in list(self.track_identities.keys()):
                        if tid not in active_track_ids:
                            del self.track_identities[tid]

                    # ── 3. Phân tích từng khuôn mặt (Liveness + ArcFace) ──
                    any_spoof_detected = False
                    authenticated_user = None
                    unknown_detected = False

                    for detection, track in tracked:
                        tid = track.track_id
                        bbox = track.bbox
                        landmarks = track.landmarks

                        # (A) Kiểm tra Liveness (Anti-Spoofing) nếu bật PAD
                        if self.enable_pad:
                            if track.needs_pad(interval_seconds=0.2):
                                try:
                                    is_real, real_prob, pad_score = self.antispoof.predict(frame, bbox)
                                    track.update_pad(is_real, pad_score)
                                except Exception:
                                    pass

                        # (B) Nhận diện Khuôn Mặt ArcFace
                        # Điều kiện nhận diện: Không PAD HOẶC Có PAD và đã xác nhận là REAL
                        should_recognize = (not self.enable_pad) or (track.pad_ready and not track.is_spoof)

                        if should_recognize and tid not in self.track_identities:
                            if landmarks is not None and len(landmarks) == 5:
                                aligned = align_face(frame, landmarks)
                                if aligned is not None:
                                    feat = self.embedder.extract(aligned)
                                    if feat is not None:
                                        matched_name, score = self.db.match_face(feat, threshold=self.args.threshold)
                                        self.track_identities[tid] = (matched_name, score)
                                        status_str = f"HỢP LỆ ({matched_name})" if matched_name else "UNKNOWN"
                                        print(f"[Nhận diện] Track #{tid}: {status_str} | Score: {score:.3f}")

                        # Đánh giá trạng thái khuôn mặt
                        if self.enable_pad and track.pad_ready and track.is_spoof:
                            any_spoof_detected = True
                        elif (not self.enable_pad) or (track.pad_ready and not track.is_spoof):
                            identity = self.track_identities.get(tid)
                            if identity and identity[0] is not None:
                                authenticated_user = identity[0]
                            else:
                                unknown_detected = True

                    # ── 4. Điều Khiển Phần Cứng Theo ĐÚNG YÊU CẦU ĐỀ BÀI ──
                    # Trường hợp 1: Phát hiện GIẢ MẠO (Ảnh / Màn hình) -> BẬT ĐÈN ĐỎ
                    if any_spoof_detected:
                        self.access_ctrl.set_spoof()
                        system_status_msg = "CẢNH BÁO: GIẢ MẠO (SPOOF) -> ĐÈN ĐỎ BẬT!"
                        system_status_color = (0, 0, 255)

                    # Trường hợp 2: NHẬN DIỆN ĐÚNG người trong CSDL -> MỞ SERVO 90 ĐỘ
                    elif authenticated_user is not None:
                        if not self.access_ctrl.is_door_open:
                            self.access_ctrl.set_authorized_open()
                            print(f"[MỞ CỬA] Chào mừng '{authenticated_user}'! -> Đã quay Servo 90 độ.")
                        system_status_msg = f"XÁC THỰC THÀNH CÔNG: {authenticated_user} -> SERVO MỞ 90°"
                        system_status_color = (0, 255, 0)

                    # Trường hợp 3: Người thật nhưng KHÔNG CÓ TRONG CSDL -> BẬT ĐÈN XANH
                    elif unknown_detected:
                        if not self.access_ctrl.is_door_open:
                            self.access_ctrl.set_unknown()
                        system_status_msg = "NGƯỜI THẬT CHƯA ĐĂNG KÝ -> ĐÈN XANH BẬT"
                        system_status_color = (255, 255, 0)

                    # Nếu có vật cản nhưng chưa bắt được mặt
                    elif len(tracked) == 0 and not self.access_ctrl.is_door_open:
                        self.access_ctrl.reset_leds()
                        system_status_msg = f"CÓ VẬT CẢN ({distance_cm:.1f} cm) - Vui lòng nhìn vào camera"
                        system_status_color = (0, 215, 255)

                else:
                    # KHÔNG CÓ VẬT CẢN PHÍA TRƯỚC:
                    # Bỏ qua nhận diện để tiết kiệm CPU, tắt đèn, đóng servo
                    if not self.access_ctrl.is_door_open:
                        self.access_ctrl.reset_leds()
                    self.track_identities.clear()
                    system_status_msg = f"CHỜ VẬT CẢN (D = {distance_cm:.1f} cm | Yêu cầu <= {self.access_ctrl.obstacle_distance_cm} cm)"
                    system_status_color = (160, 160, 160)

                # ── 5. Hiển Thị Giao Diện Hoặc Ghi Log Headless ──
                if not self.args.headless:
                    try:
                        display_frame = frame.copy()
                        h, w = frame.shape[:2]

                        # Header Banner 2 hàng
                        header_h = 60
                        cv2.rectangle(display_frame, (0, 0), (w, header_h), (25, 25, 25), -1)

                        # Hàng 1: Khoảng cách siêu âm & Trạng thái phần cứng
                        dist_color = (0, 255, 0) if has_obstacle else (180, 180, 180)
                        cv2.putText(
                            display_frame,
                            f"SIEU AM: {distance_cm:.1f} cm {'[CO VAT CAN]' if has_obstacle else '[TRONG]'}",
                            (12, 22),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.52,
                            dist_color,
                            2,
                            cv2.LINE_AA,
                        )

                        # Trạng thái Servo & LED bên phải
                        servo_str = "SERVO: 90* (MO)" if self.access_ctrl.is_door_open else "SERVO: 0* (DONG)"
                        servo_color = (0, 255, 0) if self.access_ctrl.is_door_open else (160, 160, 160)
                        (s_w, _), _ = cv2.getTextSize(servo_str, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                        cv2.putText(display_frame, servo_str, (w - s_w - 12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, servo_color, 2, cv2.LINE_AA)

                        # Hàng 2: Trạng thái hành động & Phím tắt
                        cv2.putText(display_frame, system_status_msg, (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.46, system_status_color, 2, cv2.LINE_AA)

                        fps_info = f"FPS: {fps:.1f} | [O] Test Vat Can | [Q] Thoat"
                        (f_w, _), _ = cv2.getTextSize(fps_info, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
                        cv2.putText(display_frame, fps_info, (w - f_w - 12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 170, 170), 1, cv2.LINE_AA)

                        # Đường kẻ phân cách
                        cv2.line(display_frame, (0, header_h), (w, header_h), (70, 70, 70), 1)

                        # Vẽ Bounding Box từng khuôn mặt nếu có
                        for detection, track in tracked:
                            x1, y1, x2, y2 = [int(v) for v in track.bbox]
                            identity = self.track_identities.get(track.track_id)

                            if self.enable_pad:
                                pad_val = track.last_pad_score if track.last_pad_score is not None else 0.0
                                if not track.pad_ready:
                                    box_color = (0, 255, 255)  # Vàng
                                    label = f"Kiem tra PAD... (d={pad_val:+.2f})"
                                elif track.is_spoof:
                                    box_color = (0, 0, 255)  # Đỏ
                                    label = f"SPOOF -> DEN DO (d={pad_val:+.2f})"
                                else:
                                    if identity and identity[0] is not None:
                                        box_color = (0, 255, 0)  # Xanh lục
                                        label = f"{identity[0]} -> MO SERVO 90*"
                                    else:
                                        box_color = (255, 255, 0)  # Cyan
                                        label = f"Unknown -> DEN XANH"
                            else:
                                if identity and identity[0] is not None:
                                    box_color = (0, 255, 0)
                                    label = f"{identity[0]} ({identity[1] * 100:.0f}%) -> MO SERVO"
                                else:
                                    box_color = (255, 255, 0)
                                    label = "Unknown -> DEN XANH"

                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)
                            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                            label_top = max(header_h + 4, y1 - th - 8)
                            cv2.rectangle(display_frame, (x1, label_top), (x1 + tw + 8, label_top + th + 6), box_color, -1)
                            cv2.putText(display_frame, label, (x1 + 4, label_top + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

                        cv2.imshow("Smart Access Control - Pi 4 Face Recognition", display_frame)
                        key = cv2.waitKey(1) & 0xFF

                        if key == ord("q") or key == 27:
                            print("\n[THOÁT] Đang tắt hệ thống...")
                            break
                        elif key == ord("p") or key == ord("P"):
                            self.toggle_pad_mode()
                        elif key == ord("o") or key == ord("O"):
                            self.toggle_simulate_obstacle()

                    except Exception as e:
                        print(f"[CẢNH BÁO GUI] Không thể hiển thị GUI ({e}). Chuyển sang Headless.")
                        self.args.headless = True
                else:
                    # In log định kỳ ở chế độ Console
                    if now_time - last_log_time >= 1.0:
                        obs_str = "CÓ VẬT CẢN" if has_obstacle else "TRỐNG"
                        servo_st = "MỞ 90°" if self.access_ctrl.is_door_open else "ĐÓNG 0°"
                        print(f"[LOG] Siêu âm: {distance_cm:5.1f} cm ({obs_str}) | Trạng thái: {self.access_ctrl.current_state} | Servo: {servo_st} | Số mặt: {len(tracked)} (FPS: {fps:.1f})")
                        last_log_time = now_time
                    time.sleep(0.01)

        finally:
            self.cleanup()

    def cleanup(self):
        print("[CLEANUP] Đang giải phóng camera và thiết bị ngoại vi...")
        self.access_ctrl.close()
        if self.cap.isOpened():
            self.cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("[CLEANUP] Hoàn tất an toàn. Tạm biệt!")


def main():
    args = parse_args()
    app = EdgeApp(args)

    def sig_handler(sig, frame):
        print("\n[SIGNAL] Nhận tín hiệu ngắt (Ctrl+C). Đang thoát...")
        app.running = False

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    app.run()


if __name__ == "__main__":
    main()
