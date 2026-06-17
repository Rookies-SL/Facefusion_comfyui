import importlib
import sys
import types
from pathlib import Path
from unittest import TestCase


ROOT = Path(__file__).resolve().parents[1]


class Frame:
	def __init__(self, name):
		self.name = name

	def copy(self):
		return Frame(f'{self.name}-copy')


class DummySwapper:
	def __init__(self):
		self.calls = []

	def swap_face(self, *args):
		self.calls.append(args)
		return Frame('swapped')


def install_swap_local_stubs(dummy_swapper):
	package = types.ModuleType('facefusion_api')
	package.__path__ = [str(ROOT / 'facefusion_api')]
	sys.modules['facefusion_api'] = package

	detection_package = types.ModuleType('facefusion_api.detection')
	detection_package.__path__ = [str(ROOT / 'facefusion_api' / 'detection')]
	sys.modules['facefusion_api.detection'] = detection_package

	detector_module = types.ModuleType('facefusion_api.detection.detector')
	detector_module.detect_faces = lambda *args, **kwargs: []
	sys.modules['facefusion_api.detection.detector'] = detector_module

	utils_module = types.ModuleType('facefusion_api.utils')
	utils_module.VisionFrame = Frame
	utils_module.Face = dict
	sys.modules['facefusion_api.utils'] = utils_module

	models_module = types.ModuleType('facefusion_api.models')
	models_module.get_local_swapper = lambda _model_name: dummy_swapper
	models_module.get_face_occluder = lambda *args, **kwargs: None
	models_module.get_face_parser = lambda *args, **kwargs: None
	sys.modules['facefusion_api.models'] = models_module


class SwapLocalOptimizationTest(TestCase):
	def setUp(self):
		self.dummy_swapper = DummySwapper()
		install_swap_local_stubs(self.dummy_swapper)
		sys.modules.pop('facefusion_api.swap_local', None)
		self.swap_local = importlib.import_module('facefusion_api.swap_local')

	def test_uses_precomputed_source_face_without_detecting_source_again(self):
		source_frame = Frame('source')
		target_frame = Frame('target')
		cached_source_face = {'id': 'cached-source'}
		target_face = {'id': 'target'}
		detected_frames = []

		def detect_faces(frame, *_args, **_kwargs):
			detected_frames.append(frame.name)
			return [target_face]

		self.swap_local.detect_faces = detect_faces

		result = self.swap_local.swap_faces_local(
			source_frame,
			target_frame,
			source_face=cached_source_face
		)

		self.assertEqual(result.name, 'swapped')
		self.assertEqual(detected_frames, ['target'])
		self.assertIs(self.dummy_swapper.calls[0][0], cached_source_face)
