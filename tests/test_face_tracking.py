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


def face(face_id, bbox, landmarks=None):
	result = {'id': face_id, 'bbox': bbox}
	if landmarks is not None:
		result['landmarks'] = landmarks
	return result


def landmarks_with_pose(pitch_ratio, yaw_offset=0.0, eye_distance=80.0, roll_offset=0.0):
	left_eye = [100.0, 100.0]
	right_eye = [100.0 + eye_distance, 100.0 + roll_offset]
	eye_center_x = (left_eye[0] + right_eye[0]) * 0.5
	eye_center_y = (left_eye[1] + right_eye[1]) * 0.5
	vertical_span = 80.0
	nose = [
		eye_center_x + yaw_offset * eye_distance,
		eye_center_y + pitch_ratio * vertical_span,
	]
	left_mouth = [eye_center_x - 25.0, eye_center_y + vertical_span]
	right_mouth = [eye_center_x + 25.0, eye_center_y + vertical_span]
	return [left_eye, right_eye, nose, left_mouth, right_mouth]


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

	def test_reuses_previous_face_when_candidate_jumps_to_wrong_region(self):
		previous_face = face(
			'tracked',
			[554.2, 76.7, 742.3, 251.0],
			[
				[610.0, 129.0],
				[668.0, 130.0],
				[638.0, 154.0],
				[618.0, 194.0],
				[658.0, 195.0],
			]
		)
		bad_candidate = face(
			'bad-detection',
			[574.0, 387.3, 962.5, 684.4],
			[
				[670.0, 500.0],
				[681.0, 511.0],
				[676.0, 526.0],
				[700.0, 570.0],
				[730.0, 585.0],
			]
		)

		selected = select_tracked_face([bad_candidate], previous_face, face_position=0)

		self.assertIs(selected, previous_face)

	def test_reuses_previous_face_when_landmark_pose_jumps(self):
		previous_face = face(
			'tracked',
			[543.1, 7.1, 715.2, 191.2],
			[
				[590.0, 70.0],
				[663.0, 74.0],
				[620.0, 91.0],
				[596.0, 137.0],
				[650.0, 141.0],
			]
		)
		unstable_candidate = face(
			'unstable',
			[517.8, 51.9, 705.5, 221.8],
			[
				[575.0, 95.0],
				[640.0, 102.5],
				[581.0, 111.0],
				[590.0, 167.0],
				[646.0, 173.0],
			]
		)

		selected = select_tracked_face([unstable_candidate], previous_face, face_position=0)

		self.assertIs(selected, previous_face)

	def test_reuses_previous_face_when_pitch_jumps(self):
		previous_face = face(
			'tracked',
			[520.0, 60.0, 700.0, 240.0],
			landmarks_with_pose(-0.50)
		)
		unstable_candidate = face(
			'unstable-pitch',
			[522.0, 62.0, 702.0, 242.0],
			landmarks_with_pose(-0.20)
		)

		selected = select_tracked_face([unstable_candidate], previous_face, face_position=0)

		self.assertIs(selected, previous_face)

	def test_accepts_gradual_pitch_change(self):
		previous_face = face(
			'tracked',
			[520.0, 60.0, 700.0, 240.0],
			landmarks_with_pose(-0.50)
		)
		stable_candidate = face(
			'stable-pitch',
			[522.0, 62.0, 702.0, 242.0],
			landmarks_with_pose(-0.36)
		)

		selected = select_tracked_face([stable_candidate], previous_face, face_position=0)

		self.assertIs(selected, stable_candidate)

	def test_reuses_previous_face_when_pitch_is_extreme(self):
		previous_face = face(
			'tracked',
			[520.0, 60.0, 700.0, 240.0],
			landmarks_with_pose(-0.65)
		)
		extreme_candidate = face(
			'extreme-pitch',
			[522.0, 62.0, 702.0, 242.0],
			landmarks_with_pose(-0.84)
		)

		selected = select_tracked_face([extreme_candidate], previous_face, face_position=0)

		self.assertIs(selected, previous_face)
