"""
Chương Trình Nhận Diện Khuôn Mặt & Chống Giả Mạo Tầng Biên (Pi 4 Edge Face Recognition)
Hỗ trợ 2 chế độ:
- Chế độ 1: KHÔNG PAD (Tốc độ cao 20-25 FPS, đếm số người qua hàng LED Xanh)
- Chế độ 2: CÓ PAD (Bảo mật cao, giới hạn 5 mặt gắn vào 5 cột LED: Đỏ = Spoof, Xanh = Real, Cả hai = Unknown)
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
from peripherals.led_matrix import LEDMatrix


def parse_args():
    parser = argparse.ArgumentParser(description="Pi 4 Edge Face Recognition & Anti-Spoofing")
    parser.add_argument("--pad", dest="enable_pad", action="store_true", help="Bật chế độ Chống giả mạo (PAD)")
    parser.add_argument("--no-pad", dest="enable_pad", action="store_false", help="Tắt chế độ Chống giả mạo (Không PAD)")
    parser.set_defaults(enable_pad=None)

    parser.add_argument("--camera", type=int, default=int(os.getenv("CAMERA_INDEX", "0")), help="Camera index")
    parser.add_argument("--detector", type=str, default="scrfd", help="Bộ dò khuôn mặt (mặc định: scrfd)")
    parser.add_argument("--threshold", type=float, default=float(os.getenv("MATCH_THRESHOLD", "0.62")), help="Ngưỡng so khớp ArcFace")
    parser.add_argument("--headless", action="store_true", help="Chạy không giao diện đồ họa (in log terminal)")
    return parser.parse_args()


class EdgeApp:
    def __init__(self, args):
        self.args = args

        # 1. Trạng thái PAD
        if args.enable_pad is not None:
            self.enable_pad = args.enable_pad
        else:
            self.enable_pad = os.getenv("ENABLE_PAD", "true").strip().lower() in ("true", "1", "yes")

        # Tự động phát hiện môi trường đồ họa ($DISPLAY)
        has_display = bool(os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY"))
        if not self.args.headless and not has_display:
            print("[HEADLESS TỰ ĐỘNG] Không phát hiện biến $DISPLAY (chạy qua SSH). Tự động chuyển sang chế độ Console Log & Điều khiển LED.")
            self.args.headless = True

        # 2. Khởi tạo Database
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
        print(f"[AI Modules] Nhận diện : ArcFace (buffalo_s) | Ngưỡng MATCH_THRESHOLD = {self.args.threshold}")
        print(f"[AI Modules] Anti-Spoof: {self.antispoof.model_name} | Ngưỡng FAS_LOGIT_THRESHOLD = {self.antispoof.threshold_logit}")

        # 4. Khởi tạo Phần cứng LED Matrix (10 LED)
        self.led_matrix = LEDMatrix()

        # 5. Cấu hình Camera
        self.cap = cv2.VideoCapture(args.camera)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not self.cap.isOpened():
            print(f"[LỖI] Không thể kết nối với Webcam tại /dev/video{args.camera}!")
            sys.exit(1)

        # 6. Quản lý 5 Slot Cột LED (Chế độ CÓ PAD)
        # slot_assignment: mapping track_id -> slot_index (0..4)
        self.slot_assignment = {}
        self.track_identities = {}  # track_id -> (name, score)

        self.running = True
        self.detection_interval = 0.2
        self.last_detection_time = 0.0

    def get_free_slot(self) -> int:
        """Tìm slot cột trống đầu tiên trong 5 slot (0..4). Trả về -1 nếu hết slot."""
        occupied = set(self.slot_assignment.values())
        for slot in range(5):
            if slot not in occupied:
                return slot
        return -1

    def toggle_pad_mode(self):
        """Đổi chế độ qua lại giữa CÓ PAD và KHÔNG PAD."""
        self.enable_pad = not self.enable_pad
        self.slot_assignment.clear()
        self.track_identities.clear()
        self.led_matrix.reset_all()

        mode_name = "CÓ PAD (BẢO MẬT CAO)" if self.enable_pad else "KHÔNG PAD (TỐC ĐỘ CAO)"
        print(f"\n[CHUYỂN CHẾ ĐỘ] -> {mode_name}\n")

    def run(self):
        print("\n" + "=" * 65)
        print(" HỆ THỐNG NHẬN DIỆN KHUÔN MẶT & CHỐNG GIẢ MẠO PI 4")
        print("=" * 65)
        print(f" - Chế độ ban đầu     : {'CÓ PAD (Giới hạn 5 mặt)' if self.enable_pad else 'KHÔNG PAD (Đếm số mặt)'}")
        print(f" - Bộ dò khuôn mặt    : {self.detector_type.upper()}")
        print(f" - Model Nhận diện    : ArcFace w600k_mbf (Ngưỡng so khớp: {self.args.threshold})")
        print(f" - Model Anti-Spoof   : {self.antispoof.model_name} (Ngưỡng logit: {self.antispoof.threshold_logit})")
        print(" Phím tắt điều khiển:")
        print("   [p] : Chuyển đổi giữa CÓ PAD <--> KHÔNG PAD tức thời")
        print("   [q] : Thoát chương trình")
        print("=" * 65 + "\n")

        fps = 0.0
        frame_count = 0
        fps_timer = time.time()

        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    continue

                now_time = time.time()
                frame_count += 1
                if now_time - fps_timer >= 1.0:
                    fps = frame_count / (now_time - fps_timer)
                    frame_count = 0
                    fps_timer = now_time

                # ── 1. Detection Cadence (Dò mặt cách quãng + Optical Flow) ──
                need_detect = (now_time - self.last_detection_time) >= self.detection_interval
                if need_detect or self.tracker.needs_redetection():
                    detections = self.detector.detect(frame)
                    self.last_detection_time = now_time

                    # Nếu ở chế độ CÓ PAD: giới hạn tối đa 5 khuôn mặt lớn nhất
                    if self.enable_pad and len(detections) > 5:
                        detections.sort(
                            key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]),
                            reverse=True,
                        )
                        detections = detections[:5]

                    tracked = self.tracker.update(detections, frame=frame, detector_called=True)
                else:
                    tracked = self.tracker.update([], frame=frame, detector_called=False)

                # Giới hạn 5 khuôn mặt trong chế độ CÓ PAD
                if self.enable_pad and len(tracked) > 5:
                    tracked.sort(
                        key=lambda item: (item[1].bbox[2] - item[1].bbox[0]) * (item[1].bbox[3] - item[1].bbox[1]),
                        reverse=True,
                    )
                    tracked = tracked[:5]

                active_track_ids = {t.track_id for _, t in tracked}

                # ── 2. Dọn dẹp các Slot đã rời khỏi khung hình ──
                for tid in list(self.slot_assignment.keys()):
                    if tid not in active_track_ids:
                        del self.slot_assignment[tid]
                        if tid in self.track_identities:
                            del self.track_identities[tid]

                # ── 3. Xử lý từng Khuôn Mặt & Cập nhật Trạng thái ──
                slot_statuses = {i: "EMPTY" for i in range(5)}

                for detection, track in tracked:
                    tid = track.track_id
                    bbox = track.bbox
                    landmarks = track.landmarks

                    # Gán Slot cố định cho Track ID nếu chưa có
                    if tid not in self.slot_assignment:
                        free_slot = self.get_free_slot()
                        if free_slot != -1:
                            self.slot_assignment[tid] = free_slot

                    slot_idx = self.slot_assignment.get(tid, -1)

                    # (A) Xử lý Liveness nếu ở Chế độ CÓ PAD
                    if self.enable_pad:
                        if track.needs_pad(interval_seconds=0.2):
                            try:
                                is_real, real_prob, pad_score = self.antispoof.predict(frame, bbox)
                                track.update_pad(is_real, pad_score)
                            except Exception:
                                pass

                    # (B) Nhận diện Khuôn Mặt (Trích xuất ArcFace)
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
                                    status_str = f"KHỚP ({matched_name})" if matched_name else "UNKNOWN"
                                    print(f"[Nhận diện] Track #{tid}: {status_str} | Điểm Cosine: {score:.3f} (Ngưỡng: {self.args.threshold})")

                    # (C) Xác định trạng thái của Slot này để điều khiển LED
                    if self.enable_pad and slot_idx != -1:
                        if not track.pad_ready:
                            slot_statuses[slot_idx] = "EMPTY"  # Đang tích lũy votes
                        elif track.is_spoof:
                            slot_statuses[slot_idx] = "SPOOF"
                        else:
                            # Là người thật (REAL)
                            identity = self.track_identities.get(tid)
                            if identity and identity[0] is not None:
                                slot_statuses[slot_idx] = "REAL_KNOWN"
                            else:
                                slot_statuses[slot_idx] = "UNKNOWN"

                # ── 4. Điều Khiển Phần Cứng Ma Trận LED ──
                if not self.enable_pad:
                    # Chế độ KHÔNG PAD: Đếm số lượng khuôn mặt
                    self.led_matrix.set_no_pad_mode(len(tracked))
                else:
                    # Chế độ CÓ PAD: Điều khiển 5 Cột
                    self.led_matrix.set_pad_mode(slot_statuses)

                # ── 5. Vẽ Giao Diện Hoặc Ghi Log Headless ──
                if not self.args.headless:
                    try:
                        display_frame = frame.copy()
                        h, w = frame.shape[:2]

                        # Thanh trạng thái phía trên (Header Banner 2 hàng gọn gàng, không chồng lấn)
                        header_h = 52
                        cv2.rectangle(display_frame, (0, 0), (w, header_h), (25, 25, 25), -1)

                        if self.enable_pad:
                            mode_label = "CHE DO: CO PAD (MAX 5 MAT)"
                            mode_color = (0, 215, 255)  # Vàng cam
                        else:
                            mode_label = f"CHE DO: KHONG PAD ({len(tracked)} MAT)"
                            mode_color = (0, 255, 0)  # Xanh lá

                        # Hàng 1 (y = 22): Chế độ bên trái, FPS bên phải
                        cv2.putText(display_frame, mode_label, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 2, cv2.LINE_AA)

                        fps_str = f"FPS: {fps:.1f}"
                        (fps_w, _), _ = cv2.getTextSize(fps_str, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
                        cv2.putText(display_frame, fps_str, (w - fps_w - 12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)

                        # Hàng 2 (y = 44): Phím tắt bên trái, Detector bên phải
                        hint_str = "[P] Doi Che Do  |  [Q] Thoat"
                        cv2.putText(display_frame, hint_str, (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

                        det_str = f"Detector: {self.detector_type.upper()}"
                        (det_w, _), _ = cv2.getTextSize(det_str, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                        cv2.putText(display_frame, det_str, (w - det_w - 12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA)

                        # Đường kẻ phân cách nhẹ dưới header
                        cv2.line(display_frame, (0, header_h), (w, header_h), (60, 60, 60), 1)

                        # Vẽ từng khuôn mặt
                        for detection, track in tracked:
                            tid = track.track_id
                            x1, y1, x2, y2 = [int(v) for v in track.bbox]
                            slot_idx = self.slot_assignment.get(tid, -1)
                            slot_prefix = f"[Cot {slot_idx + 1}] " if (self.enable_pad and slot_idx != -1) else ""

                            identity = self.track_identities.get(tid)

                            if self.enable_pad:
                                pad_val = track.last_pad_score if track.last_pad_score is not None else 0.0
                                if not track.pad_ready:
                                    box_color = (0, 255, 255)  # Vàng
                                    label = f"{slot_prefix}Kiem tra PAD... (d={pad_val:+.2f})"
                                elif track.is_spoof:
                                    box_color = (0, 0, 255)  # Đỏ
                                    label = f"{slot_prefix}CANH BAO: SPOOF (d={pad_val:+.2f})"
                                else:
                                    # Real
                                    if identity and identity[0] is not None:
                                        box_color = (0, 255, 0)  # Xanh
                                        label = f"{slot_prefix}{identity[0]} ({identity[1] * 100:.0f}%) [d={pad_val:+.2f}]"
                                    else:
                                        box_color = (0, 165, 255)  # Cam
                                        label = f"{slot_prefix}Unknown [d={pad_val:+.2f}]"
                            else:
                                # Không PAD
                                if identity and identity[0] is not None:
                                    box_color = (0, 255, 0)
                                    label = f"{identity[0]} ({identity[1] * 100:.0f}%)"
                                else:
                                    box_color = (0, 165, 255)
                                    label = f"Face #{tid}"

                            # Vẽ Bounding Box
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)

                            # Vẽ Nhãn Tên chuẩn xác kích thước và tránh đè lên header
                            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                            label_top = max(header_h + 4, y1 - th - 8)
                            cv2.rectangle(display_frame, (x1, label_top), (x1 + tw + 8, label_top + th + 6), box_color, -1)
                            cv2.putText(display_frame, label, (x1 + 4, label_top + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

                        cv2.imshow("Pi 4 Edge Face Recognition", display_frame)
                        key = cv2.waitKey(1) & 0xFF

                        if key == ord("q") or key == 27:
                            print("\n[THOÁT] Đang tắt hệ thống...")
                            break
                        elif key == ord("p") or key == ord("P"):
                            self.toggle_pad_mode()
                    except Exception as e:
                        print(f"[CẢNH BÁO DISPLAY] Không thể hiển thị GUI ({e}). Tự động chuyển sang Headless.")
                        self.args.headless = True
                else:
                    # Chế độ Headless: In log định kỳ ra Terminal
                    if now_time - getattr(self, "last_console_log", 0.0) >= 1.0:
                        if not self.enable_pad:
                            print(f"[KHÔNG PAD] Đang phát hiện: {len(tracked)} khuôn mặt -> Bật {min(len(tracked), 5)} LED Xanh (FPS: {fps:.1f})")
                        else:
                            active_info = []
                            for i in range(5):
                                st = slot_statuses.get(i, "EMPTY")
                                if st != "EMPTY":
                                    active_info.append(f"Cột {i+1}: {st}")
                            info_str = " | ".join(active_info) if active_info else "Đang chờ người..."
                            print(f"[CÓ PAD] {len(tracked)} mặt | {info_str} | [PAD: {self.antispoof.model_name}, Thresh: {self.antispoof.threshold_logit}] (FPS: {fps:.1f})")
                        self.last_console_log = now_time
                    time.sleep(0.01)

        finally:
            self.cleanup()

    def cleanup(self):
        print("[CLEANUP] Đang giải phóng camera và tắt 10 đèn LED...")
        self.led_matrix.close()
        if self.cap.isOpened():
            self.cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("[CLEANUP] Hoàn tất. Tạm biệt!")


def main():
    args = parse_args()
    app = EdgeApp(args)

    def sig_handler(sig, frame):
        print("\n[SIGNAL] Nhận tín hiệu ngắt (Ctrl+C). Đang thoát an toàn...")
        app.running = False

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    app.run()


if __name__ == "__main__":
    main()
