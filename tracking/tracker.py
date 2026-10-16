import time
from collections import deque
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np


def compute_iou(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
    """Tính IOU giữa 2 bounding box (x1, y1, x2, y2)."""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

    union_area = float(bbox1_area + bbox2_area - intersection_area)
    if union_area <= 0:
        return 0.0

    return float(intersection_area / union_area)


class Track:
    """Đối tượng theo dõi một khuôn mặt qua các frame."""

    def __init__(
        self,
        track_id: int,
        bbox: Tuple[int, int, int, int],
        pad_smooth_window: int = 5,
        pad_spoof_min_ratio: float = 0.6,
        pad_min_votes: Optional[int] = None,
        pad_stale_timeout: Optional[float] = None,
    ):
        self.track_id = track_id
        self.bbox = bbox
        self.landmarks = None
        self.detector_score: Optional[float] = None
        self.bbox_source = "DETECTOR"
        self.tracker_status = "detected"
        self.tracker_confidence: Optional[float] = 1.0
        self.name = "UNKNOWN"
        self.score = 0.0
        self.last_recognition_time: Optional[float] = None  # None = ép nhận diện ngay khi đủ điều kiện
        self.frames_since_recognition = 0
        self.missing_frames = 0
        self.recognized_once = False  # True sau lần update_result() đầu tiên
        self.recognition_error = False

        self.verification_generation = 0
        self.recognition_status = ""
        self.recognition_latency = 0.0
        self.recognition_eligible = False
        self.display_identity = None
        self._display_candidate = None
        self._display_candidate_count = 0
        self._display_mismatch_since = None

        # PAD temporal smoothing: lưu N verdict gần nhất (True=real, False=spoof)
        self._pad_window: deque = deque(maxlen=pad_smooth_window)
        self._pad_spoof_min_ratio = pad_spoof_min_ratio
        self.pad_min_votes = pad_min_votes if pad_min_votes is not None else pad_smooth_window
        self._pad_stale_timeout: Optional[float] = pad_stale_timeout
        self.last_pad_time: Optional[float] = None  # None = chưa chạy PAD lần nào
        self.last_pad_score: Optional[float] = None
        self.pad_invalid_reason: Optional[str] = None
        self._pad_crop_geometry = None
        self._pad_crop_time = None

        # Attendance tracking
        self.employee_id: Optional[str] = None
        self.stable_recognitions = 0
        self.last_attendance_time: Optional[float] = None
        self._unknown_streak = 0

    def update_pad(self, is_real: bool, pad_score: Optional[float] = None) -> None:
        """Ghi nhận verdict PAD của frame hiện tại vào rolling window và cập nhật timestamp.

        Nếu verdict cuối cùng đã quá cũ (>_pad_stale_timeout giây), reset window trước khi thêm
        vote mới để tránh để dữ liệu cũ làm sai lệch kết quả (ví dụ: đối tượng vừa quay lại sau khi mất khỏi frame).
        """
        if (
            self._pad_stale_timeout is not None
            and self.last_pad_time is not None
            and (time.monotonic() - self.last_pad_time) > self._pad_stale_timeout
        ):
            self._pad_window.clear()
        self._pad_window.append(bool(is_real))
        self.last_pad_time = time.monotonic()
        if pad_score is not None:
            self.last_pad_score = float(pad_score)

    def reset_pad(self) -> None:
        """Reset rolling window và đưa track về trạng thái PAD_PENDING."""
        self._pad_window.clear()
        self.last_pad_time = None
        self.last_pad_score = None
        self.pad_invalid_reason = None
        self._pad_crop_geometry = None
        self._pad_crop_time = None

    def pad_crop_bbox(self, bbox, frame_shape, expansion_factor):
        """Time-based smoothing of small crop jitter; large motion snaps immediately.

        Experimental constants: tau 0.12s, max jitter 5% of crop base side,
        reset after a 0.5s gap. Border crops bypass smoothing to preserve padding.
        Does not modify tracking geometry or PAD votes.
        """
        x1, y1, x2, y2 = map(float, bbox)
        side = max(x2 - x1, y2 - y1)
        if not np.isfinite([x1, y1, x2, y2]).all() or x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid PAD crop bbox")
        current = np.array([(x1 + x2) / 2, (y1 + y2) / 2, side])
        now = time.monotonic()
        previous = self._pad_crop_geometry

        def touches_border(geometry):
            cx, cy, size = geometry
            half = size * expansion_factor / 2
            h, w = frame_shape[:2]
            return cx - half < 0 or cy - half < 0 or cx + half > w or cy + half > h

        if previous is not None:
            dt = now - self._pad_crop_time
            center_delta = np.linalg.norm(current[:2] - previous[:2]) / side
            size_delta = abs(side - previous[2]) / side
            if (0 < dt <= 0.5 and center_delta <= 0.05 and size_delta <= 0.05
                    and not touches_border(current) and not touches_border(previous)):
                alpha = -np.expm1(-dt / 0.12)
                candidate = previous + alpha * (current - previous)
                if not touches_border(candidate):
                    current = candidate
        self._pad_crop_geometry = current
        self._pad_crop_time = now
        cx, cy, side = current
        return (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)

    def invalidate_recognition(self, retry_delay: bool = False) -> None:
        """Discard an unverified identity; failed attempts retry at normal cadence."""
        self.employee_id = None
        self.name = "UNKNOWN"
        self.score = 0.0
        self.recognized_once = False
        self.recognition_error = False
        self.stable_recognitions = 0
        self._unknown_streak = 0
        self.last_attendance_time = None
        self.last_recognition_time = time.monotonic() if retry_delay else None
        self.display_identity = None
        self._display_candidate = None
        self._display_candidate_count = 0
        self._display_mismatch_since = None

    def update_display_identity(self, identity, confirmations=2, unknown_confirmations=3):
        """UI-only hysteresis. Never modifies the raw identity/authentication state."""
        code, name, score = identity
        if self.display_identity is None:
            if code is not None:
                self.display_identity = identity
            self._display_candidate = None
            self._display_candidate_count = 0
            self._display_mismatch_since = None
            return
        if code == self.display_identity[0]:
            self.display_identity = identity
            self._display_candidate = None
            self._display_candidate_count = 0
            self._display_mismatch_since = None
            return
        if self._display_mismatch_since is None:
            self._display_mismatch_since = time.monotonic()
        candidate = code if code is not None else "__UNKNOWN__"
        if candidate == self._display_candidate:
            self._display_candidate_count += 1
        else:
            self._display_candidate = candidate
            self._display_candidate_count = 1
        required = unknown_confirmations if code is None else confirmations
        if self._display_candidate_count >= required:
            self.display_identity = identity if code is not None else None
            self._display_mismatch_since = None
            self._display_candidate = None
            self._display_candidate_count = 0

    def display_result(self, hold_seconds=1.5):
        """Expire disagreement on wall time even when the next cloud result is late."""
        stale = self._display_mismatch_since is not None
        if stale and time.monotonic() - self._display_mismatch_since >= hold_seconds:
            self.display_identity = None
        if self.display_identity is None:
            # Do not show an unconfirmed alternative identity as a confirmed label.
            return "UNKNOWN", 0.0, stale
        return self.display_identity[1], self.display_identity[2], stale

    def invalidate_verification(self) -> None:
        """Lost continuity or failed PAD requires fresh PAD and recognition."""
        self.verification_generation += 1
        self.recognition_status = ""
        self.recognition_eligible = False
        self.reset_pad()
        self.invalidate_recognition()

    def needs_pad(self, interval_seconds: float = 0.2) -> bool:
        """Kiểm tra xem track có cần chạy PAD inference lại hay không (tính theo giây)."""
        if self.last_pad_time is None:
            return True
        return (time.monotonic() - self.last_pad_time) >= interval_seconds

    @property
    def pad_ready(self) -> bool:
        """True nếu track đã tích lũy đủ số vote tối thiểu pad_min_votes và verdict chưa cũ (stale)."""
        if len(self._pad_window) < self.pad_min_votes:
            return False
        # Nếu đã vượt quá thời gian stale timeout mà chưa có update mới, coi như PAD_PENDING
        if (
            self._pad_stale_timeout is not None
            and self.last_pad_time is not None
            and (time.monotonic() - self.last_pad_time) > self._pad_stale_timeout
        ):
            return False
        return True

    @property
    def is_spoof(self) -> bool:
        """
        Trả về True nếu track đã tích lũy đủ vote VÀ tỉ lệ frame SPOOF trong rolling window >= pad_spoof_min_ratio.
        Khi chưa đủ vote (PAD_PENDING), trả về False.
        """
        if not self.pad_ready or not self._pad_window:
            return False
        spoof_ratio = self._pad_window.count(False) / len(self._pad_window)
        return spoof_ratio >= self._pad_spoof_min_ratio

    @property
    def is_real(self) -> bool:
        """True khi track đã tích lũy đủ vote VÀ phán quyết không phải là SPOOF."""
        return self.pad_ready and not self.is_spoof

    @property
    def pad_status(self) -> str:
        """
        Trạng thái PAD của track:
            - 'PAD_PENDING': chưa đủ số vote tối thiểu.
            - 'SPOOF': đủ vote và tỉ lệ spoof >= ngưỡng.
            - 'REAL': đủ vote và là người thật.
        """
        if not self.pad_ready:
            return "PAD_PENDING"
        return "SPOOF" if self.is_spoof else "REAL"

    def needs_recognition(self, interval_seconds: float = 1.0, pad_enabled: bool = True) -> bool:
        """
        Kiểm tra xem track có cần re-verify/nhận diện lại hay không (tính theo giây).
        Nếu pad_enabled=True: CHẶN nhận diện hoàn toàn nếu track chưa pad_ready hoặc là SPOOF.
        """
        if pad_enabled and (not self.pad_ready or not self.is_real):
            return False
        if self.last_recognition_time is None:
            return True
        return (time.monotonic() - self.last_recognition_time) >= interval_seconds

    @property
    def is_pending(self) -> bool:
        """True khi track chưa được nhận diện lần nào (chưa có kết quả nhận diện đầu tiên)."""
        return not self.recognized_once

    def can_log_attendance(self, gap_minutes: float) -> bool:
        """Kiểm tra xem đã hết cooldown (gap_minutes) để điểm danh lại chưa."""
        if self.last_attendance_time is None:
            return True
        return (time.time() - self.last_attendance_time) >= (gap_minutes * 60)

    def update_result(self, employee_id: Optional[str], name: str, score: float):
        """Cập nhật kết quả nhận diện và ghi nhận mốc thời gian hoàn tất."""
        if employee_id is not None:
            self._unknown_streak = 0
            if self.employee_id == employee_id:
                self.stable_recognitions += 1
            else:
                self.stable_recognitions = 1
                # Reset cooldown điểm danh nếu đổi sang một employee_id khác
                if self.employee_id is not None and self.employee_id != employee_id:
                    self.last_attendance_time = None
            self.employee_id = employee_id
        else:
            self.stable_recognitions = 0
            self._unknown_streak += 1
            # Chỉ xoá employee_id và cooldown nếu liên tục UNKNOWN > 3 chu kỳ (~1.5s)
            # Giúp tránh nháy frame UNKNOWN làm mất timer cooldown của người vừa điểm danh
            if self._unknown_streak > 3:
                self.employee_id = None
                self.last_attendance_time = None

        self.name = name
        self.score = score
        self.recognition_error = False
        self.last_recognition_time = time.monotonic()
        self.recognized_once = True


class FaceTracker:
    """
    Theo dõi khuôn mặt bằng IOU association + optical-flow propagation.

    ``update(..., detector_called=True)`` nhận detection mới và reassociate ID.
    ``update(..., detector_called=False, frame=...)`` không chạy detector mà
    propagate bbox từ frame trước bằng Lucas-Kanade optical flow. Vì vậy caller
    không được giữ nguyên bbox cũ một cách mù quáng khi detector bị skip.
    """

    def __init__(
        self,
        recognize_interval_seconds: float = 1.0,
        iou_threshold: float = 0.3,
        max_missing_frames: int = 10,
        pad_smooth_window: int = 5,
        pad_spoof_min_ratio: float = 0.6,
        pad_interval_seconds: float = 0.2,
        pad_min_votes: Optional[int] = None,
        pad_stale_timeout: Optional[float] = None,
    ):
        self.recognize_interval_seconds = recognize_interval_seconds
        self.iou_threshold = iou_threshold
        self.max_missing_frames = max_missing_frames
        self.pad_smooth_window = pad_smooth_window
        self.pad_spoof_min_ratio = pad_spoof_min_ratio
        self.pad_interval_seconds = pad_interval_seconds
        self.pad_min_votes = pad_min_votes if pad_min_votes is not None else pad_smooth_window
        self.pad_stale_timeout = pad_stale_timeout
        self.tracks: List[Track] = []
        self._next_track_id = 1
        self._previous_gray: Optional[np.ndarray] = None

    @staticmethod
    def _gray(frame) -> Optional[np.ndarray]:
        if frame is None or not hasattr(frame, "size") or frame.size == 0:
            return None
        if len(frame.shape) == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _valid_bbox(bbox, frame_shape=None) -> bool:
        if bbox is None or len(bbox) != 4:
            return False
        x1, y1, x2, y2 = [float(v) for v in bbox]
        if not all(np.isfinite(v) for v in (x1, y1, x2, y2)):
            return False
        if x2 <= x1 or y2 <= y1:
            return False
        if frame_shape is not None:
            h, w = frame_shape[:2]
            if x2 <= 0 or y2 <= 0 or x1 >= w or y1 >= h:
                return False
        return True

    @staticmethod
    def _clamp_bbox(bbox, frame_shape):
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        return (
            max(0, min(int(round(x1)), w - 1)),
            max(0, min(int(round(y1)), h - 1)),
            max(1, min(int(round(x2)), w)),
            max(1, min(int(round(y2)), h)),
        )

    def _tracker_detection(self, track: Track) -> Dict:
        return {
            "bbox": track.bbox,
            "score": track.detector_score,
            "landmarks": track.landmarks,
            "bbox_source": "TRACKER",
            "detector_called": False,
            "tracker_status": track.tracker_status,
            "tracker_confidence": track.tracker_confidence,
        }

    def _mark_all_lost(self) -> None:
        for track in self.tracks:
            track.invalidate_verification()
            track.tracker_status = "lost"
            track.tracker_confidence = 0.0
            track.missing_frames += 1

    def _propagate(self, frame) -> List[Tuple[Dict, Track]]:
        gray = self._gray(frame)
        if gray is None or self._previous_gray is None:
            self._mark_all_lost()
            self._previous_gray = gray
            return []

        results: List[Tuple[Dict, Track]] = []
        for track in self.tracks:
            track.frames_since_recognition += 1
            x1, y1, x2, y2 = track.bbox
            points = np.array(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2],
                 [(x1 + x2) / 2.0, (y1 + y2) / 2.0]],
                dtype=np.float32,
            ).reshape(-1, 1, 2)

            try:
                next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                    self._previous_gray,
                    gray,
                    points,
                    None,
                    winSize=(21, 21),
                    maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
                )
            except cv2.error:
                next_points, status = None, None

            if next_points is None or status is None:
                track.invalidate_verification()
                track.tracker_status = "lost"
                track.tracker_confidence = 0.0
                track.missing_frames += 1
                continue

            valid = status.reshape(-1).astype(bool)
            valid &= np.isfinite(next_points.reshape(-1, 2)).all(axis=1)
            valid_count = int(valid.sum())
            if valid_count < 3:
                track.invalidate_verification()
                track.tracker_status = "lost"
                track.tracker_confidence = valid_count / len(points)
                track.missing_frames += 1
                continue

            old_points = points.reshape(-1, 2)
            new_points = next_points.reshape(-1, 2)
            displacement = new_points[valid] - old_points[valid]
            dx, dy = np.median(displacement, axis=0)

            if valid_count >= 4 and valid[:4].all():
                # Dùng 4 góc để giữ translation + scale khi optical flow ổn định.
                candidate = (
                    float(np.min(new_points[:4, 0])),
                    float(np.min(new_points[:4, 1])),
                    float(np.max(new_points[:4, 0])),
                    float(np.max(new_points[:4, 1])),
                )
            else:
                width, height = x2 - x1, y2 - y1
                candidate = (x1 + dx, y1 + dy, x1 + dx + width, y1 + dy + height)

            frame_shape = gray.shape
            if not self._valid_bbox(candidate, frame_shape):
                track.invalidate_verification()
                track.tracker_status = "lost"
                track.tracker_confidence = valid_count / len(points)
                track.missing_frames += 1
                continue

            frame_h, frame_w = frame_shape[:2]
            if (
                candidate[0] < 0 or candidate[1] < 0
                or candidate[2] > frame_w or candidate[3] > frame_h
            ):
                # Request a detector refresh, not a loss of continuity. The
                # caller must redetect before inference; an unmatched detector
                # result will invalidate verification in update().
                track.tracker_status = "needs_redetection"
                track.tracker_confidence = valid_count / len(points)
                continue

            candidate = self._clamp_bbox(candidate, frame_shape)
            width, height = candidate[2] - candidate[0], candidate[3] - candidate[1]
            old_width, old_height = x2 - x1, y2 - y1
            if (
                width < 8 or height < 8
                or width > old_width * 2.5
                or height > old_height * 2.5
            ):
                track.invalidate_verification()
                track.tracker_status = "lost"
                track.tracker_confidence = valid_count / len(points)
                track.missing_frames += 1
                continue

            scale_x = width / max(old_width, 1)
            scale_y = height / max(old_height, 1)
            scale_changed = (
                scale_x < 0.8 or scale_x > 1.25
                or scale_y < 0.8 or scale_y > 1.25
            )

            if scale_changed or not valid[:4].all():
                # Keep the last detector/valid-flow geometry for association.
                # A same-frame detector match can preserve the existing votes.
                track.tracker_status = "needs_redetection"
                track.tracker_confidence = valid_count / len(points)
                continue

            track.bbox = candidate
            if track.landmarks is not None:
                # Apply the same motion (including scale/rotation), rather than
                # translating detector landmarks while the bbox changes scale.
                matrix, _ = cv2.estimateAffinePartial2D(
                    old_points[valid], new_points[valid], method=cv2.LMEDS
                )
                if matrix is None or not np.isfinite(matrix).all():
                    track.invalidate_verification()
                    track.tracker_status = "lost"
                    track.tracker_confidence = 0.0
                    track.missing_frames += 1
                    continue
                landmarks = np.asarray(track.landmarks, dtype=np.float64)
                track.landmarks = (landmarks @ matrix[:, :2].T + matrix[:, 2]).tolist()
            track.bbox_source = "TRACKER"
            track.tracker_status = "tracking"
            track.tracker_confidence = valid_count / len(points)
            track.missing_frames = 0
            results.append((self._tracker_detection(track), track))

        self._previous_gray = gray
        self.tracks = [
            t for t in self.tracks if t.missing_frames <= self.max_missing_frames
        ]
        return results

    def needs_redetection(self) -> bool:
        """Return True when propagation is unsafe or no active track exists."""
        if not self.tracks:
            return True
        return any(
            track.tracker_status not in {"tracking", "detected"}
            or not self._valid_bbox(track.bbox)
            for track in self.tracks
        )

    def update(
        self,
        detections: List[Dict],
        frame=None,
        detector_called: bool = True,
    ) -> List[Tuple[Dict, Track]]:
        """
        Khớp các detection mới với tracks hiện có.
        Trả về list các tuple (detection, track).
        """
        if not detector_called:
            return self._propagate(frame)

        matched_results: List[Tuple[Dict, Track]] = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))

        # Tăng frame counter cho tất cả track hiện có
        for track in self.tracks:
            track.frames_since_recognition += 1
            track.missing_frames += 1

        if detections and self.tracks:
            # Tính ma trận IOU giữa detections và tracks
            iou_matrix = []
            for det in detections:
                det_bbox = det["bbox"]
                row = [compute_iou(det_bbox, track.bbox) for track in self.tracks]
                iou_matrix.append(row)

            # Greedy matching theo IOU cao nhất
            used_dets = set()
            used_tracks = set()

            matches = []
            for d_idx, row in enumerate(iou_matrix):
                for t_idx, iou_val in enumerate(row):
                    if iou_val >= self.iou_threshold:
                        matches.append((iou_val, d_idx, t_idx))

            matches.sort(key=lambda x: x[0], reverse=True)

            for _, d_idx, t_idx in matches:
                if d_idx in used_dets or t_idx in used_tracks:
                    continue
                used_dets.add(d_idx)
                used_tracks.add(t_idx)

                det = detections[d_idx]
                track = self.tracks[t_idx]
                track.bbox = det["bbox"]
                track.landmarks = det.get("landmarks")
                track.detector_score = det.get("score")
                track.bbox_source = "DETECTOR"
                track.tracker_status = "detected"
                track.tracker_confidence = det.get("score", 1.0)
                track.missing_frames = 0
                matched_det = dict(det)
                matched_det["bbox_source"] = "DETECTOR"
                matched_det["detector_called"] = True
                matched_det["tracker_status"] = "detected"
                matched_det["tracker_confidence"] = track.tracker_confidence
                matched_results.append((matched_det, track))

            unmatched_detections = [i for i in range(len(detections)) if i not in used_dets]
            unmatched_tracks = [i for i in range(len(self.tracks)) if i not in used_tracks]

        # Reset chuỗi ổn định nếu track bị mất dấu ở frame này (mất focus / quay mặt)
        for t_idx in unmatched_tracks:
            self.tracks[t_idx].invalidate_verification()
            self.tracks[t_idx].tracker_status = "lost"
            self.tracks[t_idx].tracker_confidence = 0.0

        # Tạo track mới cho các detection chưa match
        for d_idx in unmatched_detections:
            det = detections[d_idx]
            new_track = Track(
                track_id=self._next_track_id,
                bbox=det["bbox"],
                pad_smooth_window=self.pad_smooth_window,
                pad_spoof_min_ratio=self.pad_spoof_min_ratio,
                pad_min_votes=self.pad_min_votes,
                pad_stale_timeout=self.pad_stale_timeout,
            )
            self._next_track_id += 1
            new_track.missing_frames = 0
            new_track.landmarks = det.get("landmarks")
            new_track.detector_score = det.get("score")
            new_track.bbox_source = "DETECTOR"
            new_track.tracker_status = "detected"
            new_track.tracker_confidence = det.get("score", 1.0)
            self.tracks.append(new_track)
            det = dict(det)
            det["bbox_source"] = "DETECTOR"
            det["detector_called"] = True
            det["tracker_status"] = "detected"
            det["tracker_confidence"] = new_track.tracker_confidence
            matched_results.append((det, new_track))

        # Xóa các track đã mất dấu quá max_missing_frames
        self.tracks = [t for t in self.tracks if t.missing_frames <= self.max_missing_frames]

        # Lưu frame detector gần nhất làm điểm bắt đầu cho optical flow.
        self._previous_gray = self._gray(frame)

        return matched_results
