import json
import logging
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

import dither_cli as cli
from pixelization import pixelize_regular
from processing_config import ConfigValidationError, validate_config


class ValidationTests(unittest.TestCase):
    def validate(self, **sections):
        return validate_config(sections, Path('job.json'), skip_input_check=True)

    def test_numeric_strings_are_normalized(self):
        result = self.validate(pixelization={'max_size': '8'}, palette={'num_colors': '4'},
                               final_resize={'multiplier': '3'})
        self.assertEqual(result['pixelization']['max_size'], 8)
        self.assertEqual(result['palette']['num_colors'], 4)
        self.assertEqual(result['final_resize']['multiplier'], 3)

    def test_bad_numbers_are_rejected(self):
        for section, field, bad_values in (
            ('pixelization', 'max_size', [0, 1, -2, True, 2.5, '1', '2.5', None]),
            ('palette', 'num_colors', [0, -1, False, 1.5, None]),
            ('final_resize', 'multiplier', [0, -1, True, 1.5, None]),
        ):
            for bad in bad_values:
                with self.subTest(section=section, value=bad):
                    with self.assertRaises(ConfigValidationError):
                        self.validate(**{section: {field: bad}})

    def test_bad_structures_have_validation_errors(self):
        for config in ([], None, {'palette': []}, {'palette': {'source': 12}},
                       {'dithering': {'parameters': []}}, {'pixelization': {'enabled': 'false'}}):
            with self.subTest(config=config):
                with self.assertRaises(ConfigValidationError):
                    validate_config(config, Path('job.json'), skip_input_check=True)

    def test_validation_does_not_mutate_original(self):
        original = {'pixelization': {'max_size': '8'}}
        validate_config(original, Path('job.json'), skip_input_check=True)
        self.assertEqual(original, {'pixelization': {'max_size': '8'}})

    def test_direct_pixelization_rejects_zero_dimension_target(self):
        with self.assertRaisesRegex(ValueError, 'at least 2'):
            pixelize_regular(Image.new('RGB', (10, 10)), 1)


class BatchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.inputs = self.root / 'inputs'
        self.inputs.mkdir()
        Image.new('RGB', (4, 4)).save(self.inputs / 'good.png')
        (self.inputs / 'bad.png').write_bytes(b'not an image')
        self.config = validate_config({'input': str(self.inputs), 'output': str(self.root / 'out'),
                                      'dithering': {'enabled': False}}, self.root / 'job.json')
        logger = patch.object(cli, 'logger', logging.getLogger('batch-tests'))
        logger.start()
        self.addCleanup(logger.stop)

    def test_partial_failure_returns_false(self):
        with patch.object(cli, 'process_single_image', side_effect=[True, False]):
            self.assertFalse(cli.process_folder(self.config))

    def test_all_success_returns_true(self):
        with patch.object(cli, 'process_single_image', return_value=True):
            self.assertTrue(cli.process_folder(self.config))

    def test_interrupt_stops_before_videos(self):
        (self.inputs / 'movie.mp4').touch()
        with patch.object(cli, 'process_single_image', side_effect=[True, KeyboardInterrupt]), \
             patch.object(cli, 'process_single_video') as video:
            with self.assertRaises(KeyboardInterrupt):
                cli.process_folder(self.config)
            video.assert_not_called()

    def test_actual_cli_partial_failure_exits_nonzero(self):
        job = self.root / 'job.json'
        job.write_text(json.dumps(self.config), encoding='utf-8')
        result = subprocess.run([sys.executable, '-X', 'utf8', 'dither_pie.py', '-q', str(job)],
                                cwd=Path(__file__).resolve().parents[1], capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertTrue((self.root / 'out' / 'good.png').exists())

    def test_video_with_dithering_disabled_does_not_generate_palette(self):
        output = self.root / 'disabled.mp4'
        self.config['output'] = str(output)
        with patch.object(cli, 'VideoProcessor') as processor, \
             patch.object(cli, 'setup_palette_from_config') as palette, \
             patch.object(cli, 'CLIProgressCallback'):
            processor.return_value.get_video_info.return_value = {
                'width': 32, 'height': 24, 'fps': 12, 'frame_count': 12}

            def process(input_path, output_path, ditherer, **kwargs):
                self.assertIsNone(ditherer)
                Path(output_path).write_bytes(b'encoded')
                return True

            processor.return_value.process_video_streaming.side_effect = process
            self.assertTrue(cli.process_single_video(self.config))
            palette.assert_not_called()
