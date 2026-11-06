"""Clip-wide fractional grids, palette locking, streaming and GIF timing."""
import json
import logging
from pathlib import Path
import pickle
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from pixel_snapper import SnapSettings
from video_snapper import VideoSnapPlan, analyze_video_snap, gif_info, is_animated_gif, _locked_grid
from video_processor import VideoProcessor, _process_frame_image


def make_gif(path):
    frames = []
    for color in ((200, 50, 80), (40, 150, 200), (220, 180, 40)):
        data = np.zeros((6, 8, 4), dtype=np.uint8)
        data[1:5, 1:7] = (*color, 255)
        frames.append(Image.fromarray(data).resize((32, 24), Image.Resampling.NEAREST))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=[20, 70, 40], loop=2, disposal=2)


class ClipPlanTests(unittest.TestCase):
    def test_auto_keeps_gradient_colors_beyond_palette_limit(self):
        y, x = np.indices((32, 32))
        source = Image.fromarray(np.stack((x * 4 + 30, y * 4 + 30, x + y + 30), axis=-1).astype(np.uint8))
        with patch.object(VideoProcessor, 'get_video_info', return_value={'frame_count': 1, 'fps': 30}), \
                patch('video_snapper.read_clip_frame', return_value=source):
            plan = analyze_video_snap('gradient.mp4', SnapSettings(pixel_size=1))
            manual = analyze_video_snap('gradient.mp4', SnapSettings(num_colors=16, pixel_size=1))
        self.assertTrue(plan.preserve_colors)
        self.assertEqual(plan.render(source).tobytes(), source.tobytes())
        self.assertEqual(len(plan.render(source).getcolors(2048)), 1024)
        self.assertFalse(manual.preserve_colors)
        self.assertLessEqual(len(manual.render(source).getcolors(2048)), 16)
        edited = plan.with_palette(((1, 2, 3),), custom=True)
        self.assertFalse(edited.preserve_colors)
        self.assertEqual(edited.render(source).getcolors(), [(1024, (1, 2, 3))])
        self.assertTrue(pickle.loads(pickle.dumps(plan)).preserve_colors)

    def test_auto_preserves_small_exact_video_palette(self):
        source = Image.new('RGB', (8, 8), (20, 30, 40))
        source.paste((50, 60, 70), (4, 0, 8, 8))
        with patch.object(VideoProcessor, 'get_video_info', return_value={'frame_count': 1, 'fps': 30}), \
                patch('video_snapper.read_clip_frame', return_value=source):
            plan = analyze_video_snap('pixelart.mp4', SnapSettings(pixel_size=1))
        self.assertFalse(plan.preserve_colors)
        self.assertEqual(set(plan.palette), {(20, 30, 40), (50, 60, 70)})
        self.assertEqual(plan.render(source).tobytes(), source.tobytes())

    def test_fractional_grid_uses_multiple_frames_and_ignores_black_frame(self):
        y, x = np.indices((18, 12))
        data = np.where(((x + y) % 2)[..., None], [210, 70, 90], [40, 140, 180]).astype(np.uint8)
        image = Image.fromarray(data).resize((64, 96), Image.Resampling.NEAREST)
        cols, rows, step = _locked_grid([image, Image.new('RGB', image.size)], SnapSettings())
        self.assertAlmostEqual(step, 16 / 3, places=2)
        self.assertEqual((len(cols) - 1, len(rows) - 1), (12, 18))

    def test_plan_is_pickleable_and_padding_keeps_logical_pixels(self):
        plan = VideoSnapPlan((12, 12), (0, 4, 8, 12), (0, 4, 8, 12), 4,
                            ((20, 60, 90), (200, 50, 80)), SnapSettings(pixel_size=4), clean=True)
        plan = pickle.loads(pickle.dumps(plan))
        source = Image.new('RGB', (12, 12), (20, 60, 90))
        source.paste((200, 50, 80), (8, 8, 12, 12))
        native = plan.render(source)
        result = _process_frame_image(source, None, 'snapper', snapper_plan=plan)
        self.assertEqual(result.size, (4, 4))
        self.assertEqual(result.crop((0, 0, 3, 3)).tobytes(), native.tobytes())
        self.assertEqual(result.getpixel((3, 3)), native.getpixel((2, 2)))
        edited = plan.with_palette(((1, 2, 3), (4, 5, 6)))
        self.assertEqual(edited.render(source).getpixel((0, 0)), (1, 2, 3))


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
class VideoSnapperTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.gif = self.root / 'source.gif'
        make_gif(self.gif)

    def test_gif_preserves_all_timing_loop_palette_and_transparency(self):
        self.assertTrue(is_animated_gif(self.gif))
        from processing_config import detect_mode
        self.assertEqual(detect_mode(self.gif), 'video')
        plan = analyze_video_snap(self.gif, SnapSettings(pixel_size=4))
        self.assertEqual(plan.output_size, (8, 6))
        self.assertEqual(set(plan.palette), {(200, 50, 80), (40, 150, 200), (220, 180, 40)})
        output = self.root / 'output.gif'
        self.assertTrue(VideoProcessor(1).process_video_streaming(str(self.gif), str(output), None, snapper_plan=plan))
        info = gif_info(output)
        self.assertEqual(info['durations'], (20, 70, 40))
        self.assertEqual(info['loop'], 2)
        self.assertEqual(info['frame_count'], 3)
        with Image.open(output) as animation:
            self.assertEqual(animation.size, (8, 6))
            for index, color in enumerate(((200, 50, 80), (40, 150, 200), (220, 180, 40))):
                animation.seek(index)
                rgba = animation.convert('RGBA')
                self.assertEqual(rgba.getpixel((0, 0))[3], 0)
                self.assertEqual(rgba.getpixel((3, 3)), (*color, 255))

    def test_gif_cancellation_preserves_destination_and_cleans_staging(self):
        plan = analyze_video_snap(self.gif, SnapSettings(pixel_size=4))
        output = self.root / 'existing.gif'
        output.write_bytes(b'keep')
        def cancel(fraction, message):
            if message.startswith('Processed 2'):
                raise RuntimeError('cancel')
        self.assertFalse(VideoProcessor(1, cancel).process_video_streaming(str(self.gif), str(output), None, snapper_plan=plan))
        self.assertEqual(output.read_bytes(), b'keep')
        self.assertEqual(list(self.root.glob('dither_*')), [])

    def test_optional_dithering_keeps_gif_timing_and_alpha(self):
        from dithering_lib import ImageDitherer, DitherMode
        plan = analyze_video_snap(self.gif, SnapSettings(pixel_size=4))
        output = self.root / 'dithered.gif'
        ditherer = ImageDitherer(palette=[(0, 0, 0), (255, 255, 255)], dither_mode=DitherMode.NONE)
        self.assertTrue(VideoProcessor(1).process_video_streaming(str(self.gif), str(output), ditherer, snapper_plan=plan))
        self.assertEqual(gif_info(output)['durations'], (20, 70, 40))
        with Image.open(output) as image:
            rgba = image.convert('RGBA')
            self.assertEqual(rgba.getpixel((0, 0))[3], 0)
            self.assertIn(rgba.getpixel((3, 3))[:3], ((0, 0, 0), (255, 255, 255)))

    def test_mp4_workers_share_grid_and_preserve_audio_frame_count(self):
        source = self.root / 'source.mp4'
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-f', 'lavfi',
            '-i', 'testsrc2=size=64x48:rate=12:duration=1', '-f', 'lavfi', '-i', 'sine=frequency=440:duration=1',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', str(source)], check=True)
        output = self.root / 'out.mp4'
        plan = analyze_video_snap(source, SnapSettings(8, pixel_size=4))
        with patch('video_snapper.analyze_video_snap', side_effect=AssertionError('Plan must be reused')):
            self.assertTrue(VideoProcessor(2).process_video_streaming(str(source), str(output), None, snapper_plan=plan))
        probe = subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-show_streams', '-of', 'json', str(output)],
                               capture_output=True, text=True, check=True)
        streams = json.loads(probe.stdout)['streams']
        video = next(s for s in streams if s['codec_type'] == 'video')
        self.assertEqual(int(video['nb_read_frames']), 12)
        self.assertEqual((video['width'], video['height']), tuple(value + value % 2 for value in plan.output_size))
        self.assertTrue(any(s['codec_type'] == 'audio' for s in streams))

    def test_cli_animated_gif_is_processed_without_dithering(self):
        import dither_cli
        from processing_config import validate_config
        output = self.root / 'cli.gif'
        config = validate_config({'input': str(self.gif), 'output': str(output),
            'pixelization': {'enabled': True, 'method': 'snapper', 'snapper': {'pixel_size': 4}},
            'dithering': {'enabled': False}}, self.root / 'job.json')
        with patch.object(dither_cli, 'logger', logging.getLogger('video-snapper-test')):
            self.assertTrue(dither_cli.process_single_video(config))
        self.assertEqual(gif_info(output)['durations'], (20, 70, 40))

    def test_transparent_gif_to_mp4_keeps_background_black(self):
        plan = analyze_video_snap(self.gif, SnapSettings(pixel_size=4))
        output = self.root / 'animation.mp4'
        self.assertTrue(VideoProcessor(2).process_video_streaming(str(self.gif), str(output), None,
                        snapper_plan=plan, final_resize_multiplier=4))
        frame = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(output), '-frames:v', '1',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], capture_output=True, check=True)
        pixels = np.frombuffer(frame.stdout, np.uint8).reshape(24, 32, 3)
        self.assertLess(int(pixels[0, 0].max()), 10)
