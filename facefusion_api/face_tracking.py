"""
Helpers for keeping the same target face selected across video frames.
"""
from typing import Any, Dict, List, Optional


Face = Dict[str, Any]


def select_tracked_face(faces: List[Face], previous_face: Optional[Face], face_position: int = 0) -> Optional[Face]:
    """Select the face that best matches the previous frame selection."""
    if not faces:
        return None

    if previous_face is None or previous_face.get('bbox') is None:
        return faces[min(face_position, len(faces) - 1)]

    previous_bbox = previous_face['bbox']
    best_face = max(
        faces,
        key=lambda face: (_bbox_iou(previous_bbox, face.get('bbox')), -_center_distance(previous_bbox, face.get('bbox')))
    )
    return best_face


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
