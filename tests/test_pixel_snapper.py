"""Grid recovery, color reduction, alpha, and image-job integration."""
from pathlib import Path
from io import BytesIO
import logging
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from pixel_snapper import SnapSettings, snap_pixels, apply_snap_palette


def enlarged_sprite(widths=(4,) * 7, heights=(4,) * 5):
    y, x = np.indices((len(heights), len(widths)))
    rgb = np.where(((x + y) % 2)[..., None], [220, 40, 70], [25, 170, 210]).astype(np.uint8)
    enlarged = np.repeat(np.repeat(rgb, heights, axis=0), widths, axis=1)
    return Image.fromarray(enlarged), Image.fromarray(rgb)


class SnapperTests(unittest.TestCase):
    def test_compressed_upscaled_art_recovers_rare_block_colors(self):
        # A tiny face/eye accent occupies very little area compared with the
        # background. A global average-error palette can erase it cheaply.
        rng = np.random.default_rng(12)
        native = np.full((48, 48, 3), (110, 130, 95), dtype=np.uint8)
        palette = np.array([[244, 238, 219], [242, 198, 198], [168, 125, 147],
                            [80, 77, 104], [200, 119, 98], [36, 43, 62]], dtype=np.uint8)
        native[12:32, 14:34] = palette[rng.integers(0, len(palette), (20, 20))]
        native[20, 22] = (74, 127, 177)
        encoded = BytesIO()
        Image.fromarray(native).resize((384, 384), Image.Resampling.NEAREST).save(
            encoded, format='JPEG', quality=85)
        encoded.seek(0)
        source = Image.open(encoded).convert('RGB')
        self.assertGreater(len(np.unique(np.asarray(source).reshape(-1, 3), axis=0)), 256)
        result = snap_pixels(source, SnapSettings(None, 'octree', 8))
        self.assertIn('Recovered approximately', result.color_note)
        self.assertEqual(result.image.size, (48, 48))
        cells = np.asarray(source).reshape(48, 8, 48, 8, 3).transpose(0, 2, 1, 3, 4)
        reference = np.rint(np.median(cells[:, :, 2:6, 2:6], axis=(2, 3)))
        distance = np.sqrt(np.sum((np.asarray(result.image).astype(float) - reference) ** 2, axis=2))
        self.assertLessEqual(float(distance.max()), 6)
        self.assertLessEqual(float(np.linalg.norm(np.array(result.image.getpixel((22, 20))) - native[20, 22].astype(float))), 8)
        self.assertLessEqual(len(result.palette), 256)

    def test_auto_color_count_preserves_a_clean_upscaled_palette_exactly(self):
        data = np.random.default_rng(8).integers(0, 256, (7, 9, 3), dtype=np.uint8)
        source = Image.fromarray(data).resize((36, 28), Image.Resampling.NEAREST)
        for algorithm in ('median_cut', 'kmeans', 'octree'):
            result = snap_pixels(source, SnapSettings(None, algorithm, 4))
            self.assertEqual(result.image.tobytes(), data.tobytes())
            self.assertEqual(len(result.palette), 63)
            self.assertIn('Preserved all 63', result.color_note)

    def test_auto_color_count_ignores_hidden_colors(self):
        data = np.random.default_rng(9).integers(0, 256, (20, 20, 4), dtype=np.uint8)
        data[..., 3] = 0
        data[:4, :4] = (12, 60, 130, 255)
        result = snap_pixels(Image.fromarray(data), SnapSettings(pixel_size=1))
        self.assertEqual(result.palette, ((12, 60, 130),))

    def test_noisy_source_uses_deterministic_estimate_and_retains_alpha(self):
        rng = np.random.default_rng(10)
        palette = np.array([[30, 70, 110], [210, 140, 80], [130, 40, 70], [60, 170, 120]])
        data = palette[np.indices((40, 40)).sum(axis=0) % 4]
        data = np.clip(data + rng.integers(-3, 4, data.shape), 0, 255).astype(np.uint8)
        rgba = np.dstack((data, np.full((40, 40), 128, dtype=np.uint8)))
        settings = SnapSettings(pixel_size=1)
        first = snap_pixels(Image.fromarray(rgba), settings)
        second = snap_pixels(Image.fromarray(rgba), settings)
        self.assertLessEqual(len(first.palette), 16)
        self.assertIn('Estimated', first.color_note)
        self.assertEqual(first.image.tobytes(), second.image.tobytes())
        self.assertEqual(first.image.getchannel('A').getextrema(), (128, 128))
        octree = snap_pixels(Image.fromarray(rgba), SnapSettings(None, 'octree', 1))
        error = np.asarray(octree.image)[..., :3].astype(float) - data
        self.assertLessEqual(float(np.sqrt(np.mean(error * error))), 3)

    def test_custom_palette_can_add_and_remove_colors_without_changing_grid(self):
        source, _ = enlarged_sprite()
        base = snap_pixels(source, SnapSettings(1, pixel_size=4))
        custom = ((220, 40, 70), (25, 170, 210))
        added = apply_snap_palette(source, base, custom, custom=True)
        self.assertEqual(set(added.image.getdata()), set(custom))
        self.assertEqual(added.columns, base.columns)
        self.assertEqual(added.rows, base.rows)
        deleted = apply_snap_palette(source, base, custom[:1], custom=True)
        self.assertEqual(set(deleted.image.getdata()), {custom[0]})
        with self.assertRaises(ValueError):
            apply_snap_palette(source, base, (), custom=True)

    def test_edited_palette_recolors_clusters_without_changing_grid_or_alpha(self):
        source, _ = enlarged_sprite()
        original = snap_pixels(source)
        edited = tuple((10, 20, 30) if i == 0 else color for i, color in enumerate(original.palette))
        result = snap_pixels(source, palette=edited)
        self.assertEqual(result.palette, edited)
        self.assertEqual(result.columns, original.columns)
        self.assertEqual(result.rows, original.rows)
        before, after = np.asarray(original.image), np.asarray(result.image)
        mask = np.all(before == original.palette[0], axis=2)
        self.assertTrue(np.all(after[mask] == (10, 20, 30)))
        np.testing.assert_array_equal(after[~mask], before[~mask])
        with self.assertRaisesRegex(ValueError, 'one RGB color'):
            snap_pixels(source, palette=((0, 0, 0),))

    def test_auto_recovers_clean_sprite_and_odd_dimensions(self):
        source, expected = enlarged_sprite()
        before = source.tobytes()
        result = snap_pixels(source)
        self.assertTrue(result.automatic)
        self.assertEqual(result.pixel_size, 4)
        self.assertEqual(result.image.size, (7, 5))
        self.assertEqual(result.image.tobytes(), expected.tobytes())
        self.assertEqual(source.tobytes(), before)

    def test_walk_follows_uneven_boundaries(self):
        widths, heights = (4, 5, 4, 3, 4, 5, 4), (4, 4, 5, 4, 3)
        source, expected = enlarged_sprite(widths, heights)
        result = snap_pixels(source, SnapSettings(pixel_size=4))
        self.assertEqual(result.image.size, expected.size)
        self.assertEqual(result.image.tobytes(), expected.tobytes())
        self.assertEqual(result.columns, tuple(np.r_[0, np.cumsum(widths)]))

    def test_dominant_cell_rejects_minority_noise(self):
        source, expected = enlarged_sprite()
        source.putpixel((2, 2), (220, 40, 70))
        result = snap_pixels(source, SnapSettings(pixel_size=4))
        self.assertEqual(result.image.tobytes(), expected.tobytes())

    def test_cell_mode_preserves_transparent_and_partial_alpha(self):
        small = Image.new('RGBA', (3, 1))
        small.putdata([(255, 0, 0, 255), (20, 90, 100, 128), (255, 0, 255, 0)])
        source = small.resize((12, 4), Image.Resampling.NEAREST)
        result = snap_pixels(source, SnapSettings(pixel_size=4)).image
        self.assertEqual(result.size, small.size)
        self.assertEqual(np.asarray(result.getchannel('A')).ravel().tolist(), [255, 128, 0])
        self.assertEqual(result.getpixel((2, 0)), (0, 0, 0, 0))

    def test_hidden_rgb_cannot_claim_palette_colors(self):
        source = Image.new('RGBA', (8, 8), (250, 0, 250, 0))
        source.paste((30, 80, 120, 255), (0, 0, 4, 4))
        for algorithm in ('median_cut', 'kmeans', 'octree'):
            result = snap_pixels(source, SnapSettings(1, algorithm, 4)).image
            self.assertEqual(result.getpixel((0, 0)), (30, 80, 120, 255))
            self.assertEqual(result.getpixel((1, 1)), (0, 0, 0, 0))

    def test_all_quantizers_limit_colors_and_are_repeatable(self):
        data = np.random.default_rng(7).integers(0, 256, (12, 15, 3), dtype=np.uint8)
        source = Image.fromarray(data)
        for algorithm in ('median_cut', 'kmeans', 'octree'):
            with self.subTest(algorithm=algorithm):
                settings = SnapSettings(7, algorithm, 1)
                first = snap_pixels(source, settings).image
                second = snap_pixels(source, settings).image
                self.assertEqual(first.size, source.size)
                self.assertLessEqual(len(first.getcolors()), 7)
                self.assertEqual(first.tobytes(), second.tobytes())

    def test_ambiguous_auto_requires_manual_override(self):
        source = Image.new('RGB', (20, 12), (50, 60, 70))
        with self.assertRaisesRegex(ValueError, 'manual pixel size'):
            snap_pixels(source)
        self.assertEqual(snap_pixels(source, SnapSettings(pixel_size=4)).image.size, (5, 3))

    def test_fully_transparent_and_palette_inputs(self):
        transparent = Image.new('RGBA', (8, 8), (200, 100, 10, 0))
        result = snap_pixels(transparent, SnapSettings(pixel_size=4)).image
        self.assertEqual(result.getchannel('A').getextrema(), (0, 0))
        source = Image.new('P', (8, 8), 0)
        source.info['transparency'] = 0
        result = snap_pixels(source, SnapSettings(pixel_size=4)).image
        self.assertEqual(result.getchannel('A').getextrema(), (0, 0))

    def test_validation_and_small_images(self):
        for kwargs in ({'num_colors': True}, {'num_colors': 0}, {'num_colors': 257},
                       {'quantizer': 'unknown'}, {'pixel_size': True}, {'pixel_size': 0},
                       {'pixel_size': float('nan')}, {'pixel_size': float('inf')}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SnapSettings(**kwargs)
        source = Image.new('RGB', (1, 1), (4, 5, 6))
        self.assertEqual(snap_pixels(source, SnapSettings(pixel_size=1)).image.tobytes(), source.tobytes())
        with self.assertRaisesRegex(ValueError, 'shorter image edge'):
            snap_pixels(source, SnapSettings(pixel_size=2))


class SnapperJobTests(unittest.TestCase):
    def setUp(self):
        self.logger = patch('dither_cli.logger', logging.getLogger('snapper-tests'))
        self.logger.start()
        self.addCleanup(self.logger.stop)

    def test_cli_image_and_normalized_settings(self):
        from processing_config import validate_config
        from dither_cli import process_single_image, generate_output_filename
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, expected = enlarged_sprite()
            source.save(root / 'in.png')
            job = {'input': 'in.png', 'output': 'out.png',
                   'pixelization': {'enabled': True, 'method': 'snapper',
                                    'snapper': {'num_colors': '16', 'pixel_size': '4', 'quantizer': 'octree'}},
                   'dithering': {'enabled': False}}
            config = validate_config(job, root / 'job.json')
            self.assertEqual(config['pixelization']['snapper']['pixel_size'], 4.0)
            self.assertEqual(job['pixelization']['snapper']['num_colors'], '16')
            self.assertTrue(process_single_image(config))
            with Image.open(root / 'out.png') as output:
                self.assertEqual(output.tobytes(), expected.tobytes())
            self.assertIn('snap_4px_16c_octree', generate_output_filename(root / 'in.png', config).name)

    def test_invalid_job_settings(self):
        from processing_config import validate_config, ConfigValidationError
        for settings in ({'pixel_size': True}, {'pixel_size': 'nan'}, {'num_colors': 257},
                         {'quantizer': 'bad'}, {'unknown': 5}, []):
            with self.subTest(settings=settings), self.assertRaises(ConfigValidationError):
                validate_config({'pixelization': {'method': 'snapper', 'snapper': settings}},
                                Path('job.json'), skip_input_check=True)

    def test_null_job_color_count_is_automatic(self):
        from processing_config import validate_config
        from dither_cli import generate_output_filename
        config = validate_config({'pixelization': {'method': 'snapper', 'enabled': True,
                                                   'snapper': {'num_colors': None}}},
                                 Path('job.json'), skip_input_check=True)
        self.assertIsNone(config['pixelization']['snapper']['num_colors'])
        self.assertIn('auto-colors', generate_output_filename(Path('in.png'), config).name)

    def test_invalid_video_fails_instead_of_silently_skipping_snapper(self):
        from dither_cli import process_single_video
        config = {'input': 'missing.mp4', 'output': 'out.mp4',
                  'pixelization': {'enabled': True, 'method': 'snapper'}}
        self.assertFalse(process_single_video(config))
