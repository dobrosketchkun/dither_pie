import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from video_processor import VideoProcessor


class VideoMetadataTests(unittest.TestCase):
    def probe(self, **overrides):
        stream = dict(width=320, height=240, avg_frame_rate='30/1', r_frame_rate='60/1')
        stream.update(overrides)
        payload = {'streams': [stream], 'format': {'duration': '12.5'}}
        with patch('video_processor.subprocess.run', return_value=SimpleNamespace(stdout=json.dumps(payload))) as run:
            result = VideoProcessor().get_video_info('input.mp4')
            self.assertEqual(run.call_count, 1)
        return result

    def test_long_duration_is_not_a_frame_count(self):
        info = self.probe(duration='150.0', nb_frames='4500')
        self.assertEqual(info['duration'], 150)
        self.assertEqual(info['frame_count'], 4500)

    def test_short_frame_count_is_not_a_duration(self):
        info = self.probe(duration='2.0', nb_frames='60')
        self.assertEqual(info['duration'], 2)
        self.assertEqual(info['frame_count'], 60)

    def test_fractional_frame_rate_is_preserved(self):
        info = self.probe(avg_frame_rate='30000/1001', duration='10.01')
        self.assertEqual(info['fps_fraction'], '30000/1001')
        self.assertEqual(info['frame_count'], 300)

    def test_missing_fields_use_named_fallbacks(self):
        info = self.probe(avg_frame_rate='0/0', r_frame_rate='24/1', duration='N/A', nb_frames='N/A')
        self.assertEqual(info['fps'], 24)
        self.assertEqual(info['duration'], 12.5)
        self.assertEqual(info['frame_count'], 300)

    def test_invalid_dimensions_or_rates_fail_explicitly(self):
        for overrides in ({'width': 0}, {'avg_frame_rate': '0/0', 'r_frame_rate': 'N/A'}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.probe(**overrides)

    def test_rotation_matches_decoded_dimensions(self):
        info = self.probe(side_data_list=[{'rotation': -90}])
        self.assertEqual((info['width'], info['height']), (240, 320))

    def test_no_video_stream_is_an_error(self):
        with patch('video_processor.subprocess.run', return_value=SimpleNamespace(stdout='{"streams": []}')):
            with self.assertRaisesRegex(ValueError, 'no video stream'):
                VideoProcessor().get_video_info('audio.mp3')
