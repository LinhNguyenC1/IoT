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

    def __init__(self, model_path: str = None, threshold_logit: float = -4.2998):
        import onnxruntime as ort

        if model_path is None:
            model_path = str(Path(__file__).resolve().parent.parent / "models" / "antispoof_minifasnet.onnx")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy model Anti-Spoofing tại: {model_path}")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.threshold_logit = threshold_logit

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

        crop = img[ty:by, lx:rx]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (size, size))

    def predict(self, frame: np.ndarray, bbox: tuple) -> tuple:
        """
        Dự đoán Real hay Spoof từ khuôn mặt trong frame.
        Trả về: (is_real: bool, real_prob: float, logit_diff: float)
        """
        cropped = self.crop_face(frame, bbox, expansion=1.5, size=80)
        if cropped is None:
            return False, 0.0, -999.0

        blob = cropped.astype(np.float32).transpose((2, 0, 1))[np.newaxis, ...]
        outputs = self.session.run(None, {self.input_name: blob})
        logits = outputs[0][0]

        # Logit 0: Spoof, Logit 1: Real
        logit_diff = float(logits[1] - logits[0])
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        real_prob = float(probs[1])
        is_real = logit_diff > self.threshold_logit

        return is_real, real_prob, logit_diff
