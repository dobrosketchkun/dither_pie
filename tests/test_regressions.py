"""Behavioral baseline for refactoring; run with unittest discovery."""

import json
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import dither_cli as cli
from dithering_lib import DitherMode, ImageDitherer, ColorReducer
from video_processor import pixelize_regular, _process_single_frame


class ImageProcessingTests(unittest.TestCase):
    def setUp(self):
        self.palette = [(0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0)]
        self.pixels = np.random.RandomState(42).randint(0, 256, (8, 8, 4)).astype(np.uint8)
        self.pixels[..., 3] = np.tile([0, 32, 64, 128, 192, 254, 255, 255], (8, 1))

    def test_all_modes_preserve_dimensions_alpha_and_palette(self):
        for mode in DitherMode:
            with self.subTest(mode=mode.value):
                params = {'size': 4} if mode == DitherMode.BLUE_NOISE else {}
                result = ImageDitherer(palette=self.palette, dither_mode=mode,
                                       dither_params=params).apply_dithering(Image.fromarray(self.pixels))
                self.assertEqual(result.size, (8, 8))
                self.assertEqual(result.mode, 'RGBA')
                np.testing.assert_array_equal(np.array(result)[..., 3], self.pixels[..., 3])
                colors = set(map(tuple, np.array(result)[..., :3].reshape(-1, 3)))
                self.assertTrue(colors.issubset(set(self.palette)), mode.value)

    def test_no_dither_uses_nearest_palette_color(self):
        source = Image.new('RGB', (3, 2), (240, 5, 10))
        result = ImageDitherer(palette=self.palette, dither_mode=DitherMode.NONE).apply_dithering(source)
        self.assertEqual(set(map(tuple, np.array(result).reshape(-1, 3))), {(255, 0, 0)})

    def test_mask_excludes_hidden_colors_from_palette(self):
        source = Image.new('RGB', (2, 1))
        source.putdata([(255, 0, 0), (0, 255, 0)])
        palette = ColorReducer.reduce_colors(source, 2, mask=np.array([[True, False]]))
        self.assertIn((255, 0, 0), palette)
        self.assertNotIn((0, 255, 0), palette)

    def test_regular_pixelization_preserves_sampled_alpha(self):
        source = Image.fromarray(self.pixels)
        result = pixelize_regular(source, 4)
        self.assertEqual(result.mode, 'RGBA')
        np.testing.assert_array_equal(np.array(result),
                                      np.array(source.resize((4, 4), Image.Resampling.NEAREST)))

    def test_frame_worker_matches_image_processing(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = Path(directory) / 'frame.png'
            source = Image.fromarray(self.pixels[..., :3])
            source.save(frame)
            ditherer = ImageDitherer(palette=self.palette)
            expected = ditherer.apply_dithering(pixelize_regular(source, 4))
            self.assertTrue(_process_single_frame(frame, ditherer, 'regular', 4))
            with Image.open(frame) as actual:
                np.testing.assert_array_equal(np.array(actual), np.array(expected))

    def test_frame_worker_reports_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(_process_single_frame(Path(directory) / 'missing.png', ImageDitherer()))


class ConfigurationAndCLITests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / 'source.png'
        Image.new('RGBA', (8, 8), (210, 20, 30, 128)).save(self.source)
        self.config_path = self.root / 'settings.json'
        self.log_patch = patch.object(cli, 'logger', logging.getLogger('tests'))
        self.log_patch.start()
        self.addCleanup(self.log_patch.stop)

    def config(self, **overrides):
        return cli.validate_config(dict(input='source.png', output='result.png', **overrides), self.config_path)

    def test_defaults_and_relative_paths(self):
        config = self.config()
        self.assertEqual(config['input'], str(self.source.resolve()))
        self.assertEqual(config['output'], str((self.root / 'result.png').resolve()))
        self.assertFalse(config['pixelization']['enabled'])
        self.assertEqual(config['dithering']['mode'], 'bayer')
        self.assertEqual(config['palette']['num_colors'], 16)

    def test_invalid_mode_rejected(self):
        with self.assertRaises(cli.ConfigValidationError):
            self.config(dithering={'mode': 'nonexistent'})

    def test_invalid_json_has_validation_error(self):
        self.config_path.write_text('{invalid', encoding='utf-8')
        with self.assertRaises(cli.ConfigValidationError):
            cli.load_config(self.config_path)

    def test_load_config_and_detect_modes(self):
        self.config_path.write_text(json.dumps({'input': 'source.png', 'output': 'result.png'}), encoding='utf-8')
        self.assertEqual(cli.load_config(self.config_path)['input'], str(self.source.resolve()))
        self.assertEqual(cli.detect_mode(self.source), 'image')
        self.assertEqual(cli.detect_mode(self.root / 'movie.MP4'), 'video')
        self.assertEqual(cli.detect_mode(self.root), 'folder')

    def test_cli_image_preserves_alpha_and_applies_resize(self):
        config = self.config(pixelization={'enabled': True, 'max_size': 4},
                             dithering={'enabled': False},
                             final_resize={'enabled': True, 'multiplier': 3})
        self.assertTrue(cli.process_single_image(config))
        with Image.open(config['output']) as result:
            self.assertEqual(result.size, (12, 12))
            self.assertEqual(result.mode, 'RGBA')
            self.assertEqual(result.getpixel((0, 0)), (210, 20, 30, 128))

    def test_cli_refuses_loss_of_transparency(self):
        config = self.config(dithering={'enabled': False})
        config['output'] = str(self.root / 'result.jpg')
        self.assertFalse(cli.process_single_image(config))
        self.assertFalse(Path(config['output']).exists())


if __name__ == '__main__':
    unittest.main()
