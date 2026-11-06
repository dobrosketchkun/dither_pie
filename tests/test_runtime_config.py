"""Native CPU detection still works when WMIC is absent."""
import os
import sys
import unittest
from unittest.mock import patch

from runtime_config import configure_windows_cpu_detection, windows_physical_cpu_count


@unittest.skipUnless(sys.platform == 'win32', 'Windows CPU API')
class WindowsCPUDetectionTests(unittest.TestCase):
    def test_native_count_avoids_wmic_and_respects_worker_limit(self):
        from joblib.externals.loky.backend import context
        count = windows_physical_cpu_count()
        self.assertIsInstance(count, int)
        self.assertGreater(count, 0)
        self.assertLessEqual(count, os.cpu_count())
        with patch.object(context, 'physical_cores_cache', None), \
                patch.object(context.subprocess, 'run', side_effect=AssertionError('No WMIC process')), \
                patch.dict(os.environ, {'LOKY_MAX_CPU_COUNT': '1'}):
            configure_windows_cpu_detection()
            self.assertEqual(context.physical_cores_cache, count)
            self.assertEqual(context.cpu_count(only_physical_cores=True), 1)
            with patch.dict(os.environ, {'LOKY_MAX_CPU_COUNT': str(os.cpu_count())}):
                self.assertEqual(context._count_physical_cores(), (count, None))

