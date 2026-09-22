import unittest
from PIL import Image
from ui.pixelization_editor import sample_editor_grid


class EditorSamplingTests(unittest.TestCase):
    def test_grid_uses_captured_transform_and_alpha_cutoff(self):
        source = Image.new('RGBA', (4, 4), (20, 40, 60, 255))
        source.putpixel((1, 1), (255, 0, 0, 128))
        source.putpixel((3, 1), (1, 2, 3, 129))
        grid = sample_editor_grid(source, 2, 2, (0, 0, 1), (0, 0, 4, 4, 2), 128)
        self.assertEqual(grid, [[None, (1, 2, 3)], [(20, 40, 60), (20, 40, 60)]])

    def test_grid_cells_outside_source_remain_absent(self):
        source = Image.new('RGBA', (2, 2), (20, 40, 60, 255))
        grid = sample_editor_grid(source, 2, 2, (0, 0, 1), (0, 0, 4, 4, 2), 0)
        self.assertEqual(grid, [[(20, 40, 60), None], [None, None]])
