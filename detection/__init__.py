"""
Detection Package Factory
Hỗ trợ cả 2 mô hình:
- 'yunnet' : OpenCV YuNet INT8 (detector_yunnet.onnx ~120 KB) - Siêu nhẹ, mặc định tối ưu cho Pi 4
- 'scrfd'  : InsightFace SCRFD (detector_scrfd.onnx ~2.5 MB) - Độ chính xác cao
"""

import os
from .yunnet_detector import FaceDetector as YuNetDetector
from .scrfd_detector import SCRFDDetector


def create_detector(detector_type: str = None, input_size=(640, 480), conf_thresh: float = 0.5):
    """
    Factory tạo detector dựa theo cấu hình.
    detector_type: 'yunnet' (mặc định) hoặc 'scrfd'
    """
    if detector_type is None:
        detector_type = os.getenv("APP_DETECTOR", "scrfd").strip().lower()

    if detector_type == "scrfd":
        try:
            return SCRFDDetector(det_size=(640, 640), conf_thresh=conf_thresh)
        except Exception as e:
            print(f"[WARN] Không thể khởi tạo SCRFD ({e}), tự động chuyển sang YuNet...")
            return YuNetDetector(input_size=input_size, score_threshold=conf_thresh)
    else:
        return YuNetDetector(input_size=input_size, score_threshold=conf_thresh)


__all__ = ["create_detector", "YuNetDetector", "SCRFDDetector"]
