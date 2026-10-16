"""
Module Phát hiện khuôn mặt YuNet (cv2.FaceDetectorYN)
Mô hình ONNX INT8 tối ưu hóa chạy trên CPU của Raspberry Pi 4.
"""

import os
import cv2
import numpy as np
from pathlib import Path


class FaceDetector:
    """YuNet Face Detector siêu nhẹ trên OpenCV DNN."""

    def __init__(self, model_path: str = None, input_size=(640, 480), score_threshold=0.5, nms_threshold=0.3):
        if model_path is None:
            model_path = str(Path(__file__).resolve().parent.parent / "models" / "detector_yunnet.onnx")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy model YuNet tại: {model_path}")

        self.detector = cv2.FaceDetectorYN.create(
            model_path, "", input_size, score_threshold=score_threshold, nms_threshold=nms_threshold, top_k=5000
        )
        self.current_size = input_size

    def detect(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        if (w, h) != self.current_size:
            self.detector.setInputSize((w, h))
            self.current_size = (w, h)

        _, faces = self.detector.detect(frame)
        if faces is None or len(faces) == 0:
            return []

        results = []
        for face in faces:
            bbox = (int(face[0]), int(face[1]), int(face[0] + face[2]), int(face[1] + face[3]))
            score = float(face[14])
            landmarks = [
                [float(face[4]), float(face[5])],    # Mắt phải
                [float(face[6]), float(face[7])],    # Mắt trái
                [float(face[8]), float(face[9])],    # Đỉnh mũi
                [float(face[10]), float(face[11])],  # Khóe miệng phải
                [float(face[12]), float(face[13])],  # Khóe miệng trái
            ]
            results.append({"bbox": bbox, "score": score, "landmarks": landmarks})
        return results
