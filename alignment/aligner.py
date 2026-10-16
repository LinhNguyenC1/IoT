"""
Module Căn chỉnh khuôn mặt (Face Alignment)
Căn chỉnh khuôn mặt chuẩn 5 điểm theo template ArcFace (112x112).
"""

import cv2
import numpy as np

# Template 5 điểm ArcFace chuẩn kích thước 112x112:
# Mắt phải, mắt trái, mũi, khóe miệng phải, khóe miệng trái
ARCFACE_DST_112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def align_face(image: np.ndarray, landmarks: list, output_size=(112, 112)) -> np.ndarray:
    """Căn chỉnh khuôn mặt dựa trên ma trận Affine 5 điểm mốc."""
    if not landmarks or len(landmarks) != 5:
        return None

    src = np.asarray(landmarks, dtype=np.float32)
    matrix, _ = cv2.estimateAffinePartial2D(src, ARCFACE_DST_112)
    if matrix is None:
        return None

    aligned = cv2.warpAffine(
        image,
        matrix,
        output_size,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return aligned
