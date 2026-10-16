"""
Detection Package
InsightFace SCRFD (detector_scrfd.onnx ~2.5 MB) - Bộ dò khuôn mặt chuẩn tốc độ cao & độ chính xác cao.
"""

from .scrfd_detector import SCRFDDetector


def create_detector(det_size=(640, 640), conf_thresh: float = 0.5):
    """
    Factory tạo bộ dò khuôn mặt SCRFD chuẩn.
    """
    return SCRFDDetector(det_size=det_size, conf_thresh=conf_thresh)


__all__ = ["create_detector", "SCRFDDetector"]
