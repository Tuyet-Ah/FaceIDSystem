"""Unicode overlay regression checks without opening camera or application DB."""
import ast
import os
import unittest
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

source = Path(__file__).resolve().parents[1] / 'backend' / 'server.py'
tree = ast.parse(source.read_text(encoding='utf-8-sig'))
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in ('camera_unicode_font', 'draw_camera_label')]
exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'))


class CameraFontTests(unittest.TestCase):
    def test_vietnamese_accents_are_not_replaced_with_question_marks(self):
        frame = np.zeros((100, 640, 3), dtype=np.uint8)
        other = frame.copy()
        draw_camera_label(frame, 'Đặng Thị Hương', (10, 50), (0, 255, 0))
        draw_camera_label(other, '??ng Th? H??ng', (10, 50), (0, 255, 0))
        self.assertTrue(frame.any())
        self.assertFalse(np.array_equal(frame, other))
        self.assertGreater(frame[:, :, 1].sum(), frame[:, :, 0].sum())

    def test_long_label_clips_safely_at_frame_edges(self):
        frame = np.zeros((30, 40, 3), dtype=np.uint8)
        draw_camera_label(frame, 'Nguyễn ' * 20, (39, 1), (255, 255, 255))
        self.assertEqual(frame.shape, (30, 40, 3))


if __name__ == '__main__':
    unittest.main()
