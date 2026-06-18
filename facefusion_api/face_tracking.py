"""
Helpers for keeping the same target face selected across video frames.
"""
from typing import Any, Dict, List, Optional

from .face_pose import estimate_face_pose


Face = Dict[str, Any]

MAX_CENTER_DISTANCE_SCALE = 0.75
MIN_EYE_DISTANCE = 35.0
MIN_EYE_DISTANCE_RATIO = 0.55
MAX_EYE_DISTANCE_RATIO = 1.65
MAX_ABS_PITCH_RATIO = 0.80
MAX_PITCH_DELTA = 0.25
MAX_YAW_OFFSET = 0.75
MAX_YAW_DELTA = 0.30
MAX_ROLL_DELTA = 20.0


def select_tracked_face(faces: List[Face], previous_face: Optional[Face], face_position: int = 0) -> Optional[Face]:
    """Select the face that best matches the previous frame selection."""
    if not faces:
        return None

    if previous_face is None or previous_face.get('bbox') is None:
        return faces[min(face_position, len(faces) - 1)]

    previous_bbox = previous_face['bbox']
    ranked_faces = sorted(
        faces,
        key=lambda face: (_bbox_iou(previous_bbox, face.get('bbox')), -_center_distance(previous_bbox, face.get('bbox'))),
        reverse=True
    )
    for candidate_face in ranked_faces:
        if _is_stable_candidate(previous_face, candidate_face):
            return candidate_face

    return previous_face


def _is_stable_candidate(previous_face: Face, candidate_face: Face) -> bool:
    previous_bbox = previous_face.get('bbox')
    candidate_bbox = candidate_face.get('bbox')
    if previous_bbox is None or candidate_bbox is None:
        return False

    if _center_distance(previous_bbox, candidate_bbox) > _max_center_distance(previous_bbox):
        return False

    previous_pose = estimate_face_pose(previous_face)
    candidate_pose = estimate_face_pose(candidate_face)
    if previous_pose.get('valid') != 1.0 or candidate_pose.get('valid') != 1.0:
        return True

    candidate_eye_distance = candidate_pose['eye_distance']
    previous_eye_distance = previous_pose['eye_distance']
    if candidate_eye_distance < MIN_EYE_DISTANCE:
        return False

    eye_distance_ratio = candidate_eye_distance / max(previous_eye_distance, 1e-6)
    if eye_distance_ratio < MIN_EYE_DISTANCE_RATIO or eye_distance_ratio > MAX_EYE_DISTANCE_RATIO:
        return False

    if abs(candidate_pose['pitch_ratio']) > MAX_ABS_PITCH_RATIO:
        return False

    pitch_delta = abs(candidate_pose['pitch_ratio'] - previous_pose['pitch_ratio'])
    if pitch_delta > MAX_PITCH_DELTA:
        return False

    if abs(candidate_pose['yaw_offset']) > MAX_YAW_OFFSET:
        return False

    yaw_delta = abs(candidate_pose['yaw_offset'] - previous_pose['yaw_offset'])
    if yaw_delta > MAX_YAW_DELTA:
        return False

    roll_delta = abs(candidate_pose['roll_degrees'] - previous_pose['roll_degrees'])
    if roll_delta > MAX_ROLL_DELTA:
        return False

    return True


def _max_center_distance(bbox) -> float:
    x1, y1, x2, y2 = map(float, bbox)
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    diagonal = (width ** 2 + height ** 2) ** 0.5
    return max(80.0, diagonal * MAX_CENTER_DISTANCE_SCALE)


def _bbox_iou(first_bbox, second_bbox) -> float:
    if first_bbox is None or second_bbox is None:
        return 0.0

    first_x1, first_y1, first_x2, first_y2 = map(float, first_bbox)
    second_x1, second_y1, second_x2, second_y2 = map(float, second_bbox)

    intersection_x1 = max(first_x1, second_x1)
    intersection_y1 = max(first_y1, second_y1)
    intersection_x2 = min(first_x2, second_x2)
    intersection_y2 = min(first_y2, second_y2)

    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height

    first_area = max(0.0, first_x2 - first_x1) * max(0.0, first_y2 - first_y1)
    second_area = max(0.0, second_x2 - second_x1) * max(0.0, second_y2 - second_y1)
    union_area = first_area + second_area - intersection_area

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def _center_distance(first_bbox, second_bbox) -> float:
    if first_bbox is None or second_bbox is None:
        return float('inf')

    first_x1, first_y1, first_x2, first_y2 = map(float, first_bbox)
    second_x1, second_y1, second_x2, second_y2 = map(float, second_bbox)
    first_center_x = (first_x1 + first_x2) * 0.5
    first_center_y = (first_y1 + first_y2) * 0.5
    second_center_x = (second_x1 + second_x2) * 0.5
    second_center_y = (second_y1 + second_y2) * 0.5

    return ((first_center_x - second_center_x) ** 2 + (first_center_y - second_center_y) ** 2) ** 0.5
