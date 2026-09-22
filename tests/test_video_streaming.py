import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from dithering_lib import ImageDitherer
from video_processor import VideoProcessor, _frame_batches, _read_frame
import video_processor


class FailsAfterFirstFrame:
    def __init__(self):
        self.calls = 0

    def apply_dithering(self, image):
        self.calls += 1
        if self.calls > 1:
            raise ValueError('Intentional worker failure')
        return image


class RawFrameTests(unittest.TestCase):
    def test_truncated_frame_is_not_silently_dropped(self):
        with self.assertRaisesRegex(ValueError, 'Truncated'):
            _read_frame(io.BytesIO(b'12345'), 6)

    def test_short_pipe_reads_are_assembled(self):
        class ShortReads(io.BytesIO):
            def read(self, count):
                return super().read(min(count, 2))
        stream = ShortReads(b'abcdef')
        self.assertEqual(_read_frame(stream, 6), b'abcdef')
        self.assertIsNone(_read_frame(stream, 6))

    def test_batches_are_bounded_and_include_final_partial_batch(self):
        batches = list(_frame_batches(io.BytesIO(b'abcdefghijklmn'), 2, 3))
        self.assertEqual(list(map(len, batches)), [3, 3, 1])
        self.assertEqual(b''.join(frame for batch in batches for frame in batch), b'abcdefghijklmn')


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
class VideoStreamingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.source = cls.root / 'source.mp4'
        subprocess.run([
            'ffmpeg', '-nostdin', '-v', 'error', '-y', '-f', 'lavfi',
            '-i', 'testsrc2=size=32x24:rate=12:duration=1',
            '-f', 'lavfi', '-i', 'sine=frequency=440:duration=1',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac',
            '-shortest', str(cls.source),
        ], capture_output=True, check=True, timeout=30)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def probe(self, path):
        result = subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-show_streams',
                                 '-of', 'json', str(path)], capture_output=True, check=True, timeout=30)
        return json.loads(result.stdout)['streams']

    def test_real_stream_preserves_frame_count_audio_and_dimensions(self):
        output = self.root / 'processed.mp4'
        progress = []
        processor = VideoProcessor(num_workers=2, progress_callback=lambda f, m: progress.append((f, m)))
        ditherer = ImageDitherer(palette=[(0, 0, 0), (255, 255, 255)])
        with patch('video_processor.Pool', wraps=video_processor.Pool) as pool, \
             patch.object(Image.Image, 'save', side_effect=AssertionError('No PNG frame extraction')):
            self.assertTrue(processor.process_video_streaming(str(self.source), str(output), ditherer,
                            pixelize_func=('regular', 12), batch_size=2))
            self.assertEqual(pool.call_count, 1)
        streams = self.probe(output)
        video = next(s for s in streams if s['codec_type'] == 'video')
        self.assertEqual(int(video['nb_read_frames']), 12)
        self.assertEqual((video['width'], video['height']), (16, 12))
        self.assertEqual(video['avg_frame_rate'], '12/1')
        audio = next(s for s in streams if s['codec_type'] == 'audio')
        self.assertEqual(audio['codec_name'], 'aac')
        self.assertAlmostEqual(float(audio['duration']), 1.0, delta=0.1)
        self.assertEqual(progress[-1][0], 1.0)
        self.assertEqual(list(self.root.glob('dither_*')), [])

    def test_worker_failure_preserves_destination_and_cleans_processes(self):
        output = self.root / 'existing.mp4'
        output.write_bytes(b'existing output')
        processes = []
        original = subprocess.Popen

        def start(*args, **kwargs):
            process = original(*args, **kwargs)
            processes.append(process)
            return process

        with patch('video_processor.subprocess.Popen', side_effect=start):
            success = VideoProcessor(num_workers=2).process_video_streaming(
                str(self.source), str(output), FailsAfterFirstFrame(), batch_size=2)
        self.assertFalse(success)
        self.assertEqual(output.read_bytes(), b'existing output')
        self.assertTrue(all(process.poll() is not None for process in processes))
        self.assertEqual(list(self.root.glob('dither_*')), [])

    def test_interrupt_cleans_children_and_keeps_destination(self):
        output = self.root / 'interrupted.mp4'
        output.write_bytes(b'original')

        def cancel(fraction, message):
            if message.startswith('Processed'):
                raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            VideoProcessor(num_workers=1, progress_callback=cancel).process_video_streaming(
                str(self.source), str(output), None, batch_size=2)
        self.assertEqual(output.read_bytes(), b'original')
        self.assertEqual(list(self.root.glob('dither_*')), [])

    def test_fractional_rate_and_silent_input(self):
        source, output = self.root / 'fractional.mp4', self.root / 'fractional_out.mp4'
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-f', 'lavfi',
                        '-i', 'testsrc2=size=32x24:rate=24000/1001', '-frames:v', '5',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(source)],
                       capture_output=True, check=True, timeout=30)
        self.assertTrue(VideoProcessor(num_workers=1).process_video_streaming(str(source), str(output), None))
        video = self.probe(output)[0]
        self.assertEqual(video['avg_frame_rate'], '24000/1001')
        self.assertEqual(int(video['nb_read_frames']), 5)

    def test_neural_path_reuses_one_model_without_worker_pool(self):
        output = self.root / 'neural_mock.mp4'
        with patch('video_processor.NeuralPixelizer') as model, patch('video_processor.Pool') as pool:
            model.return_value.pixelize.side_effect = lambda image, size: image.resize((16, 12))
            self.assertTrue(VideoProcessor(num_workers=2).process_video_streaming(
                str(self.source), str(output), None, pixelize_func=('neural', 12), batch_size=2))
            self.assertEqual(model.call_count, 1)
            self.assertEqual(model.return_value.pixelize.call_count, 12)
            pool.assert_not_called()

    def test_subtitle_stream_is_copied(self):
        subtitles = self.root / 'captions.srt'
        subtitles.write_text('1\n00:00:00,000 --> 00:00:00,900\nExample caption\n', encoding='utf-8')
        source, output = self.root / 'captions.mkv', self.root / 'captions_out.mkv'
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(self.source),
                        '-i', str(subtitles), '-map', '0', '-map', '1', '-c', 'copy', str(source)],
                       capture_output=True, check=True, timeout=30)
        self.assertTrue(VideoProcessor(num_workers=1).process_video_streaming(str(source), str(output), None))
        streams = self.probe(output)
        self.assertTrue(any(s['codec_type'] == 'subtitle' and s['codec_name'] == 'subrip' for s in streams))

    def test_odd_input_dimensions_encode_without_explicit_resize(self):
        source, output = self.root / 'odd.mkv', self.root / 'odd_out.mp4'
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-f', 'lavfi',
                        '-i', 'testsrc=size=31x23:rate=12', '-frames:v', '2',
                        '-c:v', 'ffv1', str(source)], capture_output=True, check=True, timeout=30)
        self.assertTrue(VideoProcessor(num_workers=1).process_video_streaming(str(source), str(output), None))
        video = self.probe(output)[0]
        self.assertEqual((video['width'], video['height']), (32, 24))
        self.assertEqual(int(video['nb_read_frames']), 2)

    def test_rotated_input_has_correct_output_orientation(self):
        source, output = self.root / 'rotated.mp4', self.root / 'rotated_out.mp4'
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(self.source),
                        '-c', 'copy', '-metadata:s:v:0', 'rotate=90', str(source)],
                       capture_output=True, check=True, timeout=30)
        processor = VideoProcessor(num_workers=1)
        self.assertTrue(processor.process_video_streaming(str(source), str(output), None))
        info = processor.get_video_info(str(output))
        self.assertEqual((info['width'], info['height']), (24, 32))
        video = next(s for s in self.probe(output) if s['codec_type'] == 'video')
        self.assertEqual((video['width'], video['height']), (24, 32))

    def test_variable_rate_input_is_resampled_without_duration_drift(self):
        source, output = self.root / 'variable.mp4', self.root / 'variable_out.mp4'
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(self.source),
                        '-vf', "select='lt(n,6)+gte(n,6)*not(mod(n,2))'", '-vsync', 'vfr',
                        '-an', '-c:v', 'libx264', '-bf', '0', str(source)],
                       capture_output=True, check=True, timeout=30)
        processor = VideoProcessor(num_workers=1)
        original = processor.get_video_info(str(source))
        self.assertTrue(processor.process_video_streaming(str(source), str(output), None))
        result = processor.get_video_info(str(output))
        self.assertEqual(result['fps_fraction'], original['fps_fraction'])
        self.assertAlmostEqual(result['duration'], original['duration'], delta=1 / original['fps'])

    def test_decoding_follows_timestamps_even_with_inaccurate_duration_metadata(self):
        # Older FFmpeg builds report too-short durations for this VFR/B-frame
        # combination. Read to decoder EOF rather than limiting by that estimate.
        source, output = self.root / 'reordered.mp4', self.root / 'reordered_out.mp4'
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(self.source),
                        '-vf', "select='lt(n,6)+gte(n,6)*not(mod(n,2))'", '-vsync', 'vfr',
                        '-an', '-c:v', 'libx264', str(source)],
                       capture_output=True, check=True, timeout=30)
        frames = json.loads(subprocess.check_output([
            'ffprobe', '-v', 'error', '-show_frames', '-select_streams', 'v:0',
            '-show_entries', 'frame=best_effort_timestamp_time,pkt_duration_time',
            '-of', 'json', str(source)], timeout=30))['frames']
        end_time = max(float(f['best_effort_timestamp_time']) + float(f['pkt_duration_time']) for f in frames)
        processor = VideoProcessor(num_workers=1)
        self.assertTrue(processor.process_video_streaming(str(source), str(output), None))
        result = processor.get_video_info(str(output))
        self.assertAlmostEqual(result['duration'], end_time, delta=1 / result['fps'])
