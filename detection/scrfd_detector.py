"""
Module Phát hiện khuôn mặt SCRFD (InsightFace)
Sử dụng mô hình detector_scrfd.onnx (~2.5 MB).
"""

import os
from pathlib import Path
import numpy as np

try:
    import onnxruntime as ort
    ort.set_default_logger_severity(3)
except ImportError:
    pass


class SCRFDDetector:
    """SCRFD Face Detector từ InsightFace."""

    def __init__(
        self,
        model_path: str = None,
        ctx_id: int = -1,
        det_size=(640, 640),
        conf_thresh: float = 0.5,
        min_face_size: int = 40,
    ):
        if model_path is None:
            model_path = str(Path(__file__).resolve().parent.parent / "models" / "detector_scrfd.onnx")

        self.min_face_size = max(int(min_face_size), 0)
        self._model = None

        try:
            from insightface.model_zoo import get_model
            self._model = get_model(model_path, providers=["CPUExecutionProvider"])
            self._model.prepare(ctx_id=ctx_id, input_size=det_size, det_thresh=conf_thresh)
        except Exception as e:
            try:
                from insightface.app import FaceAnalysis
                app = FaceAnalysis(name="buffalo_s", allowed_modules=["detection"], providers=["CPUExecutionProvider"])
                app.prepare(ctx_id=ctx_id, det_size=det_size, det_thresh=conf_thresh)
                self._model = app.models.get("detection")
            except Exception as e2:
                raise RuntimeError(
                    f"Không thể khởi tạo SCRFD Detector: {e}. Vui lòng kiểm tra môi trường hoặc file models/detector_scrfd.onnx"
                )

        if self._model is None:
            raise RuntimeError(f"Không tìm thấy mô hình SCRFD tại: {model_path}")

    def detect(self, frame: np.ndarray):
        bboxes, kpss = self._model.detect(frame, max_num=0, metric="default")
        if bboxes is None or len(bboxes) == 0:
            return []

        detections = []
        for i in range(bboxes.shape[0]):
            x1, y1, x2, y2, score = bboxes[i]
            if self.min_face_size > 0 and ((x2 - x1) < self.min_face_size or (y2 - y1) < self.min_face_size):
                continue
            landmarks = kpss[i].tolist() if kpss is not None and len(kpss) > i else []
            detections.append({
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "score": float(score),
                "landmarks": landmarks,
            })
        return detections
