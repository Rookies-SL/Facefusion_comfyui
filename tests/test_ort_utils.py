import importlib
import sys
import types
from io import StringIO
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class FakeSession:
	def get_providers(self):
		return ['CPUExecutionProvider']

	def get_provider_options(self):
		return {'CPUExecutionProvider': {}}


class OrtUtilsTest(TestCase):
	def setUp(self):
		package = types.ModuleType('facefusion_api')
		package.__path__ = [str(ROOT / 'facefusion_api')]
		sys.modules['facefusion_api'] = package
		sys.modules.pop('facefusion_api.ort_utils', None)

	def test_skips_provider_diagnostics_by_default(self):
		ort_utils = importlib.import_module('facefusion_api.ort_utils')
		output = StringIO()

		with patch.dict('os.environ', {}, clear=True), patch('sys.stdout', output):
			ort_utils.log_onnx_session(
				'LocalFaceSwapper:hyperswap_1c_256',
				['CUDAExecutionProvider', 'CPUExecutionProvider'],
				FakeSession()
			)

		self.assertEqual(output.getvalue(), '')

	def test_logs_available_requested_actual_providers_and_options_when_enabled(self):
		ort_utils = importlib.import_module('facefusion_api.ort_utils')
		ort_utils.ort.get_available_providers = lambda: [
			'TensorrtExecutionProvider',
			'CUDAExecutionProvider',
			'CPUExecutionProvider'
		]

		output = StringIO()

		with patch.dict('os.environ', {'FACEFUSION_ONNX_DEBUG': '1'}), patch('sys.stdout', output):
			ort_utils.log_onnx_session(
				'LocalFaceSwapper:hyperswap_1c_256',
				['CUDAExecutionProvider', 'CPUExecutionProvider'],
				FakeSession()
			)

		log = output.getvalue()
		self.assertIn('[ONNXRuntime][LocalFaceSwapper:hyperswap_1c_256] available providers:', log)
		self.assertIn('TensorrtExecutionProvider', log)
		self.assertIn("requested providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']", log)
		self.assertIn("actual providers: ['CPUExecutionProvider']", log)
		self.assertIn("provider options: {'CPUExecutionProvider': {}}", log)
