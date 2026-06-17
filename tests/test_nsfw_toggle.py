import importlib
import sys
import types
from pathlib import Path
from unittest import TestCase


ROOT = Path(__file__).resolve().parents[1]


def install_stub_modules() -> None:
	package = types.ModuleType('facefusion_api')
	package.__path__ = [str(ROOT / 'facefusion_api')]
	sys.modules['facefusion_api'] = package

	nodes_package = types.ModuleType('facefusion_api.nodes')
	nodes_package.__path__ = [str(ROOT / 'facefusion_api' / 'nodes')]
	sys.modules['facefusion_api.nodes'] = nodes_package

	comfy_module = types.ModuleType('comfy')
	comfy_types_module = types.ModuleType('comfy.comfy_types')
	comfy_types_module.IO = types.SimpleNamespace(IMAGE='IMAGE', VIDEO='VIDEO')
	sys.modules['comfy'] = comfy_module
	sys.modules['comfy.comfy_types'] = comfy_types_module

	video_types_module = types.ModuleType('comfy_api.input_impl.video_types')
	video_types_module.VideoFromComponents = object
	sys.modules['comfy_api'] = types.ModuleType('comfy_api')
	sys.modules['comfy_api.input_impl'] = types.ModuleType('comfy_api.input_impl')
	sys.modules['comfy_api.input_impl.video_types'] = video_types_module

	comfy_api_util_module = types.ModuleType('comfy_api.util')
	comfy_api_util_module.VideoComponents = object
	sys.modules['comfy_api.util'] = comfy_api_util_module

	comfy_api_nodes_util_module = types.ModuleType('comfy_api_nodes.util')
	comfy_api_nodes_util_module.bytesio_to_image_tensor = lambda output_buffer: output_buffer
	comfy_api_nodes_util_module.tensor_to_bytesio = lambda tensor, mime_type='image/webp': tensor
	sys.modules['comfy_api_nodes'] = types.ModuleType('comfy_api_nodes')
	sys.modules['comfy_api_nodes.util'] = comfy_api_nodes_util_module

	retries_module = types.ModuleType('httpx_retries')
	retries_module.Retry = object
	retries_module.RetryTransport = object
	sys.modules['httpx_retries'] = retries_module

	types_module = types.ModuleType('facefusion_api.types')
	types_module.FaceSwapperModel = str
	types_module.InputTypes = dict
	sys.modules['facefusion_api.types'] = types_module

	utils_module = types.ModuleType('facefusion_api.utils')
	utils_module.tensor_to_cv2 = lambda tensor: tensor
	utils_module.cv2_to_tensor = lambda frame: frame
	utils_module.get_average_embedding = lambda *args, **kwargs: None
	utils_module.implode_pixel_boost = lambda *args, **kwargs: None
	utils_module.explode_pixel_boost = lambda *args, **kwargs: None
	sys.modules['facefusion_api.utils'] = utils_module

	detection_module = types.ModuleType('facefusion_api.detection')
	detection_module.detect_faces = lambda *args, **kwargs: []
	detection_module.select_faces = lambda *args, **kwargs: []
	sys.modules['facefusion_api.detection'] = detection_module

	swap_local_module = types.ModuleType('facefusion_api.swap_local')
	swap_local_module.swap_faces_local = lambda **kwargs: kwargs['target_image']
	sys.modules['facefusion_api.swap_local'] = swap_local_module

	models_module = types.ModuleType('facefusion_api.models')
	models_module.get_local_swapper = lambda *args, **kwargs: None
	models_module.get_face_occluder = lambda *args, **kwargs: None
	models_module.get_face_parser = lambda *args, **kwargs: None
	models_module.MODEL_CONFIGS = {}
	sys.modules['facefusion_api.models'] = models_module


class NsfwToggleTest(TestCase):
	def setUp(self) -> None:
		install_stub_modules()
		sys.modules.pop('facefusion_api.nodes.base', None)
		sys.modules.pop('facefusion_api.nodes.image_nodes', None)
		self.image_nodes = importlib.import_module('facefusion_api.nodes.image_nodes')

	def test_swap_face_skips_nsfw_analysis_when_toggle_is_disabled(self) -> None:
		calls = []

		def analyse_frame(_frame):
			raise AssertionError('analyse_frame should not be called')

		def swap_faces_local(**kwargs):
			calls.append(kwargs)
			return 'swapped-cv2'

		self.image_nodes.tensor_to_cv2 = lambda tensor: f'cv2:{tensor}'
		self.image_nodes.cv2_to_tensor = lambda frame: f'tensor:{frame}'
		self.image_nodes.analyse_frame = analyse_frame
		self.image_nodes.swap_faces_local = swap_faces_local

		result = self.image_nodes.SwapFaceImage.swap_face(
			'source',
			'target',
			'-1',
			'hyperswap_1c_256',
			enable_nsfw_check=False
		)

		self.assertEqual(result, 'tensor:swapped-cv2')
		self.assertEqual(len(calls), 1)
