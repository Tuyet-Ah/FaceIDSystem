"""Exercise camera controls with synthetic frames; no camera or database access."""
import ast
import threading
import time
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np


class LocalCameraTests(unittest.TestCase):
    def run_camera(self, fail_switch=False, fail_open=False):
        source = Path(__file__).resolve().parents[1] / 'backend' / 'server.py'
        tree = ast.parse(source.read_text(encoding='utf-8-sig'))
        worker = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'local_camera_worker')
        cv = MagicMock()
        capture = cv.VideoCapture.return_value
        capture.isOpened.return_value = not fail_open
        capture.read.side_effect = lambda: (True, np.zeros((240, 320, 3), dtype=np.uint8))
        cv.waitKey.side_effect = [ord('2'), -1, ord('q')]
        old, new = MagicMock(), MagicMock()
        old.recognize_frame.return_value = []
        new.recognize_frame.return_value = []
        registry = MagicMock()
        def prepare(name):
            if name == 'buffalo_s' and fail_switch:
                raise RuntimeError('missing checkpoint')
            return (new if name == 'buffalo_s' else old), {}
        registry.prepare.side_effect = prepare
        state = {}
        scope = dict(
            os=SimpleNamespace(name='posix', getenv=lambda key, default: default),
            cv2=cv, web_models=registry, update_camera_state=lambda **kw: state.update(kw),
            camera_status_snapshot=lambda: dict(state), camera_stop_event=threading.Event(),
            CAMERA_INDEX=0, CAMERA_FRAME_WIDTH=320, CAMERA_FRAME_HEIGHT=240,
            CAMERA_MIRROR=False, CAMERA_SCAN_EVERY_FRAMES=1, SHOW_METRICS=True,
            SHOW_CAMERA_WINDOW=True, CAMERA_WINDOW_TITLE='test',
            MODEL_IDS=('buffalo_l', 'buffalo_s', 'adaface', 'ghostfacenet'),
            FaceValidationError=ValueError, time=time, datetime=datetime,
        )
        exec(compile(ast.Module(body=[worker], type_ignores=[]), str(source), 'exec'), scope)
        scope['local_camera_worker']()
        return state, old, new, capture

    def test_keyboard_switch_updates_model_and_uses_new_recognizer(self):
        state, old, new, capture = self.run_camera()
        self.assertEqual(state['model'], 'buffalo_s')
        self.assertEqual(old.recognize_frame.call_count, 1)
        self.assertEqual(new.recognize_frame.call_count, 1)
        self.assertEqual(state['state'], 'STOPPED')
        capture.release.assert_called_once()

    def test_failed_switch_keeps_previous_model(self):
        state, old, new, _ = self.run_camera(fail_switch=True)
        self.assertEqual(state['model'], 'buffalo_l')
        self.assertEqual(old.recognize_frame.call_count, 2)
        new.recognize_frame.assert_not_called()

    def test_unavailable_camera_reports_error(self):
        state, old, _, _ = self.run_camera(fail_open=True)
        self.assertEqual(state['state'], 'ERROR')
        self.assertFalse(state['running'])
        old.recognize_frame.assert_not_called()


if __name__ == '__main__':
    unittest.main()
