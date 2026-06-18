import importlib
import sys
import types
from pathlib import Path
from unittest import TestCase

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def install_swapper_stubs():
    package = types.ModuleType('facefusion_api')
    package.__path__ = [str(ROOT / 'facefusion_api')]
    sys.modules['facefusion_api'] = package

    models_package = types.ModuleType('facefusion_api.models')
    models_package.__path__ = [str(ROOT / 'facefusion_api' / 'models')]
    sys.modules['facefusion_api.models'] = models_package

    ort_utils_module = types.ModuleType('facefusion_api.ort_utils')
    ort_utils_module.log_onnx_session = lambda *args, **kwargs: None
    sys.modules['facefusion_api.ort_utils'] = ort_utils_module

    utils_module = types.ModuleType('facefusion_api.utils')
    utils_module.VisionFrame = np.ndarray
    utils_module.Face = dict
    utils_module.get_model_path = lambda model_name: model_name
    utils_module.ensure_model_exists = lambda *args, **kwargs: True
    utils_module.implode_pixel_boost = lambda *args, **kwargs: []
    utils_module.explode_pixel_boost = lambda *args, **kwargs: None
    sys.modules['facefusion_api.utils'] = utils_module

    onnxruntime_module = types.ModuleType('onnxruntime')
    onnxruntime_module.InferenceSession = object
    sys.modules['onnxruntime'] = onnxruntime_module

    onnx_module = types.ModuleType('onnx')
    onnx_module.numpy_helper = types.SimpleNamespace(to_array=lambda value: value)
    onnx_module.load = lambda _path: None
    sys.modules['onnx'] = onnx_module


class AffineModeTest(TestCase):
    def setUp(self):
        install_swapper_stubs()
        sys.modules.pop('facefusion_api.models.swapper', None)
        self.swapper_module = importlib.import_module('facefusion_api.models.swapper')

    def test_full_affine_mode_uses_full_affine_estimator(self):
        swapper = self.swapper_module.LocalFaceSwapper()
        landmarks = np.array([
            [10.0, 10.0],
            [20.0, 10.0],
            [15.0, 15.0],
            [12.0, 22.0],
            [18.0, 22.0],
        ], dtype=np.float32)
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        matrix = np.array([[1.0, 0.1, 2.0], [0.2, 1.0, 3.0]], dtype=np.float32)
        calls = []

        def estimate_full(source, target, method=None):
            calls.append(('full', source.copy(), target.copy(), method))
            return matrix, None

        def estimate_partial(*_args, **_kwargs):
            raise AssertionError('partial estimator should not be used')

        self.swapper_module.cv2.estimateAffine2D = estimate_full
        self.swapper_module.cv2.estimateAffinePartial2D = estimate_partial
        self.swapper_module.cv2.warpAffine = lambda *_args, **_kwargs: image

        _crop, affine_matrix = swapper._warp_face(
            image,
            landmarks,
            'arcface_128',
            (128, 128),
            affine_mode='full'
        )

        self.assertEqual(calls[0][0], 'full')
        np.testing.assert_allclose(affine_matrix, matrix)

    def test_full_affine_mode_falls_back_to_partial_when_estimation_fails(self):
        swapper = self.swapper_module.LocalFaceSwapper()
        landmarks = np.array([
            [10.0, 10.0],
            [20.0, 10.0],
            [15.0, 15.0],
            [12.0, 22.0],
            [18.0, 22.0],
        ], dtype=np.float32)
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        matrix = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]], dtype=np.float32)
        calls = []

        self.swapper_module.cv2.estimateAffine2D = lambda *_args, **_kwargs: (None, None)

        def estimate_partial(source, target, method=None):
            calls.append(('partial', source.copy(), target.copy(), method))
            return matrix, None

        self.swapper_module.cv2.estimateAffinePartial2D = estimate_partial
        self.swapper_module.cv2.warpAffine = lambda *_args, **_kwargs: image

        _crop, affine_matrix = swapper._warp_face(
            image,
            landmarks,
            'arcface_128',
            (128, 128),
            affine_mode='full'
        )

        self.assertEqual(calls[0][0], 'partial')
        np.testing.assert_allclose(affine_matrix, matrix)
