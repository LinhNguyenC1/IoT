"""
Module Dự đoán Anti-Spoofing (MiniFASNetV2)
Sử dụng ONNX Runtime CPU với mô hình antispoof_minifasnet.onnx (~1.7 MB).
"""

import os
import cv2
import numpy as np
from pathlib import Path


class AntiSpoofPredictor:
    """MiniFASNet R7 Anti-Spoofing Predictor sử dụng ONNX Runtime."""

    def __init__(self, model_path: str = None, threshold_logit: float = -0.3730):
        import onnxruntime as ort

        if model_path is None:
            env_path = os.getenv("FAS_MODEL_PATH")
            if env_path:
                p = Path(env_path)
                if not p.is_absolute():
                    p = Path(__file__).resolve().parent.parent / p
                model_path = str(p)
            else:
                model_path = str(Path(__file__).resolve().parent.parent / "models" / "R7_SC_100K_crop15.onnx")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy model Anti-Spoofing tại: {model_path}")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.model_path = model_path
        self.model_name = Path(model_path).name
        self.threshold_logit = threshold_logit
        print(f"[AntiSpoof] Đã tải model: {self.model_name} (Ngưỡng logit: {self.threshold_logit})")

    @staticmethod
    def crop_face(img: np.ndarray, bbox: tuple, expansion: float = 1.5, size: int = 80):
        src_h, src_w = img.shape[:2]
        x1, y1, x2, y2 = map(float, bbox)
        box_w, box_h = x2 - x1, y2 - y1
        if box_w <= 0 or box_h <= 0 or src_w < 2 or src_h < 2:
            return None

        used_scale = min((src_h - 1) / box_h, (src_w - 1) / box_w, float(expansion))
        new_w, new_h = box_w * used_scale, box_h * used_scale
        cx, cy = box_w / 2 + x1, box_h / 2 + y1
        left, top = cx - new_w / 2, cy - new_h / 2
        right, bottom = cx + new_w / 2, cy + new_h / 2

        if left < 0:
            right -= left
            left = 0
        if top < 0:
            bottom -= top
            top = 0
        if right > src_w - 1:
            left -= right - src_w + 1
            right = src_w - 1
        if bottom > src_h - 1:
            top -= bottom - src_h + 1
            bottom = src_h - 1

        lx, ty, rx, by = map(int, (left, top, right, bottom))
        if lx < 0 or ty < 0 or rx >= src_w or by >= src_h or rx < lx or by < ty:
            return None

        patch = img[ty : by + 1, lx : rx + 1]
        if patch.size == 0:
            return None
        return cv2.resize(patch, (size, size))

    def predict(self, frame: np.ndarray, bbox: tuple):
        """
        Dự đoán ảnh là REAL hay SPOOF.
        Trả về: (is_real: bool, real_probability: float, pad_score: float)
        """
        crop = self.crop_face(frame, bbox, expansion=1.5, size=80)
        if crop is None:
            return False, 0.0, 0.0

        # MiniFASNet nhận BGR, scale [0, 1], shape (1, 3, 80, 80)
        blob = (crop.astype(np.float32) / 255.0).transpose((2, 0, 1))[None, :]
        logits = self.session.run(None, {self.input_name: blob})[0].flatten()
        # Trong MiniFASNet R7 (R7_SC_100K_crop15), logits[0] là Real và logits[1] là Spoof
        real_logit, spoof_logit = float(logits[0]), float(logits[1])
        pad_score = real_logit - spoof_logit
        is_real = pad_score >= self.threshold_logit

        # Độ tin cậy liveness chuẩn hóa theo sigmoid quanh ngưỡng quyết định threshold_logit
        delta = pad_score - self.threshold_logit
        calibrated_score = float(1.0 / (1.0 + np.exp(-delta)))

        return is_real, calibrated_score, pad_score
