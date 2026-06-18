import importlib
import sys
import types
from io import StringIO
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class FacePoseTest(TestCase):
    def setUp(self):
        package = types.ModuleType('facefusion_api')
        package.__path__ = [str(ROOT / 'facefusion_api')]
        sys.modules['facefusion_api'] = package
        sys.modules.pop('facefusion_api.face_pose', None)
        self.face_pose = importlib.import_module('facefusion_api.face_pose')

    def test_formats_pose_diagnostics_from_five_landmarks(self):
        face = {
            'bbox': np.array([10.0, 20.0, 110.0, 160.0]),
            'landmarks': np.array([
                [35.0, 60.0],
                [85.0, 62.0],
                [62.0, 92.0],
                [45.0, 125.0],
                [80.0, 126.0],
            ], dtype=np.float32),
        }

        message = self.face_pose.format_face_pose_log('AdvancedSwapFaceVideo', 7, face)

        self.assertIn('[FacePose][AdvancedSwapFaceVideo][frame=7]', message)
        self.assertIn('bbox=(10.0,20.0,110.0,160.0)', message)
        self.assertIn('pitch_ratio=', message)
        self.assertIn('yaw_offset=', message)
        self.assertIn('roll_deg=', message)

    def test_logs_only_when_enabled_by_argument_or_environment(self):
        face = {
            'bbox': [0.0, 0.0, 10.0, 10.0],
            'landmarks': np.array([
                [1.0, 2.0],
                [9.0, 2.0],
                [5.0, 5.0],
                [3.0, 8.0],
                [7.0, 8.0],
            ], dtype=np.float32),
        }
        output = StringIO()

        with patch.dict('os.environ', {}, clear=True), patch('sys.stdout', output):
            self.face_pose.log_face_pose('AdvancedSwapFaceVideo', 1, face, enabled=False)

        self.assertEqual(output.getvalue(), '')

        with patch.dict('os.environ', {'FACEFUSION_POSE_DEBUG': '1'}), patch('sys.stdout', output):
            self.face_pose.log_face_pose('AdvancedSwapFaceVideo', 2, face, enabled=False)

        self.assertIn('[FacePose][AdvancedSwapFaceVideo][frame=2]', output.getvalue())
