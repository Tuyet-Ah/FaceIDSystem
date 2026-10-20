import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from face_service import FaceService, FaceValidationError


class HeifImagesTest(unittest.TestCase):
    def test_heif_is_decoded_to_bgr_before_face_detection(self):
        register_heif_opener()
        data = io.BytesIO()
        Image.new("RGB", (200, 180), (240, 20, 10)).save(data, format="HEIF")
        service = FaceService(Path("unused.pkl"))
        with patch.object(service, "_detect_faces", return_value=[]) as detect:
            with self.assertRaises(FaceValidationError):
                service.inspect_image(data.getvalue(), strict_quality=False)
        frame = detect.call_args.args[0]
        self.assertEqual(frame.shape, (180, 200, 3))
        np.testing.assert_allclose(frame[90, 100], [10, 20, 240], atol=10)

    def test_invalid_image_returns_validation_error(self):
        with self.assertRaises(FaceValidationError):
            FaceService(Path("unused.pkl")).inspect_image(b"invalid image")


if __name__ == "__main__":
    unittest.main()
