import importlib
import sys
import types
from pathlib import Path
from unittest import TestCase


ROOT = Path(__file__).resolve().parents[1]


package = types.ModuleType('facefusion_api')
package.__path__ = [str(ROOT / 'facefusion_api')]
sys.modules['facefusion_api'] = package

select_tracked_face = importlib.import_module('facefusion_api.face_tracking').select_tracked_face


def face(face_id, bbox):
	return {'id': face_id, 'bbox': bbox}


class FaceTrackingTest(TestCase):
	def test_keeps_same_face_by_bbox_overlap_instead_of_position(self):
		previous_face = face('left', [10, 10, 50, 50])
		current_faces = [
			face('right', [100, 10, 140, 50]),
			face('left', [12, 11, 52, 51]),
		]

		selected = select_tracked_face(current_faces, previous_face, face_position=0)

		self.assertEqual(selected['id'], 'left')

	def test_falls_back_to_position_without_previous_face(self):
		current_faces = [
			face('right', [100, 10, 140, 50]),
			face('left', [12, 11, 52, 51]),
		]

		selected = select_tracked_face(current_faces, None, face_position=1)

		self.assertEqual(selected['id'], 'left')

	def test_returns_none_when_no_faces_are_detected(self):
		self.assertIsNone(select_tracked_face([], {'bbox': [0, 0, 1, 1]}, face_position=0))
