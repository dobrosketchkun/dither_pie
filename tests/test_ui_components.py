"""Exercise GUI helpers without opening windows."""

import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

from gui_components import ZoomableImage, StatusBar, PixelizationEditorCanvas


class GUIHelperTests(unittest.TestCase):
    def test_transparency_composites_on_checkerboard(self):
        source = Image.new('RGBA', (16, 16), (255, 0, 0, 0))
        result = ZoomableImage._composite_for_display(source)
        self.assertEqual(result.mode, 'RGB')
        np.testing.assert_array_equal(np.array(result),
                                      np.array(ZoomableImage._make_checkerboard(16, 16)))

    def test_opaque_display_preserves_pixels(self):
        source = Image.new('RGB', (4, 4), (12, 34, 56))
        result = ZoomableImage._composite_for_display(source)
        np.testing.assert_array_equal(np.array(result), np.array(source))

    def test_spinner_loads_application_resource(self):
        path = Path(__file__).resolve().parents[1] / 'spinners.json'
        spinners = json.loads(path.read_text(encoding='utf-8'))
        name, config = next(iter(spinners.items()))
        status = SimpleNamespace()
        StatusBar._load_spinner(status, name)
        self.assertEqual(status.spinner_frames, config['frames'])
        self.assertEqual(status.spinner_interval, config['interval'])

    def test_editor_history_copies_rows(self):
        pixels = [[(1, 2, 3), None], [None, (4, 5, 6)]]
        saved = PixelizationEditorCanvas._deep_copy_pixels(pixels)
        pixels[0][0] = (255, 255, 255)
        self.assertEqual(saved[0][0], (1, 2, 3))


if __name__ == '__main__':
    unittest.main()
