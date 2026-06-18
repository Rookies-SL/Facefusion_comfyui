"""
Lightweight pose diagnostics for video face-swap alignment.
"""
import math
import os
from typing import Any, Dict, Optional

import numpy as np


TRUE_VALUES = {'1', 'true', 'yes', 'on'}


def is_pose_debug_enabled(enabled: bool = False) -> bool:
    """Return whether pose diagnostics should be printed."""
    if enabled:
        return True
    return os.environ.get('FACEFUSION_POSE_DEBUG', '').strip().lower() in TRUE_VALUES


def estimate_face_pose(face: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Estimate simple 2D pose metrics from five landmarks."""
    if not face or face.get('landmarks') is None:
        return {'valid': 0.0}

    landmarks = np.asarray(face['landmarks'], dtype=np.float32)
    if landmarks.shape[0] < 5:
        return {'valid': 0.0}

    left_eye, right_eye, nose, left_mouth, right_mouth = landmarks[:5]
    eye_center = (left_eye + right_eye) * 0.5
    mouth_center = (left_mouth + right_mouth) * 0.5
    eye_vector = right_eye - left_eye
    eye_distance = float(np.linalg.norm(eye_vector))
    vertical_span = float(abs(mouth_center[1] - eye_center[1]))

    safe_eye_distance = max(eye_distance, 1e-6)
    safe_vertical_span = max(vertical_span, 1e-6)
    pitch_ratio = float((nose[1] - eye_center[1]) / safe_vertical_span)
    yaw_offset = float((nose[0] - eye_center[0]) / safe_eye_distance)
    roll_degrees = float(math.degrees(math.atan2(eye_vector[1], eye_vector[0])))

    return {
        'valid': 1.0,
        'eye_distance': eye_distance,
        'pitch_ratio': pitch_ratio,
        'yaw_offset': yaw_offset,
        'roll_degrees': roll_degrees,
    }


def _format_bbox(face: Dict[str, Any]) -> str:
    bbox = face.get('bbox')
    if bbox is None:
        return 'none'
    bbox_array = np.asarray(bbox, dtype=np.float32).reshape(-1)
    if bbox_array.shape[0] < 4:
        return 'invalid'
    return '({:.1f},{:.1f},{:.1f},{:.1f})'.format(*bbox_array[:4])


def format_face_pose_log(label: str, frame_index: int, face: Optional[Dict[str, Any]]) -> str:
    """Format one pose diagnostics log line."""
    metrics = estimate_face_pose(face)
    if metrics.get('valid') != 1.0:
        return f'[FacePose][{label}][frame={frame_index}] no valid landmarks'

    return (
        f'[FacePose][{label}][frame={frame_index}] '
        f'bbox={_format_bbox(face)} '
        f'eye_dist={metrics["eye_distance"]:.2f} '
        f'pitch_ratio={metrics["pitch_ratio"]:.3f} '
        f'yaw_offset={metrics["yaw_offset"]:.3f} '
        f'roll_deg={metrics["roll_degrees"]:.2f}'
    )


def log_face_pose(label: str, frame_index: int, face: Optional[Dict[str, Any]], enabled: bool = False) -> None:
    """Print pose diagnostics when enabled."""
    if is_pose_debug_enabled(enabled):
        print(format_face_pose_log(label, frame_index, face))
