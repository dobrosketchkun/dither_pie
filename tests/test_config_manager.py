from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest

from config_manager import ConfigManager


class PreferenceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_first_run_saves_valid_defaults_without_error(self):
        path = self.root / 'preferences.json'
        output = io.StringIO()
        with redirect_stdout(output):
            manager = ConfigManager(path)
        self.assertEqual(output.getvalue(), '')
        self.assertEqual(json.loads(path.read_text(encoding='utf-8')), manager.config)
        self.assertEqual(manager.config, ConfigManager.DEFAULT_CONFIG)

    def test_new_instances_do_not_share_nested_defaults(self):
        defaults = deepcopy(ConfigManager.DEFAULT_CONFIG)
        first = ConfigManager(self.root / 'first.json')
        second = ConfigManager(self.root / 'second.json')
        first.set('window', 'width', value=999)
        first.add_recent_file('example.png')
        self.assertEqual(second.config, defaults)
        self.assertEqual(ConfigManager.DEFAULT_CONFIG, defaults)

    def test_existing_settings_load_without_changing_defaults_or_file(self):
        defaults = deepcopy(ConfigManager.DEFAULT_CONFIG)
        path = self.root / 'existing.json'
        content = '{"window": {"width": 900}, "recent_files": ["saved.png"]}'
        path.write_text(content, encoding='utf-8')
        manager = ConfigManager(path)
        self.assertEqual(manager.get('window', 'width'), 900)
        self.assertEqual(manager.get('window', 'height'), defaults['window']['height'])
        self.assertEqual(path.read_text(encoding='utf-8'), content)
        self.assertEqual(ConfigManager.DEFAULT_CONFIG, defaults)

    def test_empty_file_recovers_in_memory_without_overwriting_it(self):
        # The old first-run bug could leave an empty file behind.
        path = self.root / 'empty.json'
        path.write_text('', encoding='utf-8')
        with redirect_stdout(io.StringIO()):
            manager = ConfigManager(path)
        self.assertEqual(manager.config, ConfigManager.DEFAULT_CONFIG)
        manager.set('window', 'width', value=900)
        self.assertNotEqual(ConfigManager.DEFAULT_CONFIG['window']['width'], 900)
        self.assertEqual(path.read_text(encoding='utf-8'), '')

    def test_preferences_round_trip_after_saving(self):
        path = self.root / 'preferences.json'
        manager = ConfigManager(path)
        manager.set('defaults', 'num_colors', value=8)
        manager.save()
        loaded = ConfigManager(path)
        self.assertEqual(loaded.get('defaults', 'num_colors'), 8)

    def test_modern_parameters_and_multiplier_survive_reload(self):
        path = self.root / 'preferences.json'
        manager = ConfigManager(path)
        parameters = {'bayer': {'matrix_size': '8x8'}, 'blue-noise': {'seed': 23}}
        manager.set('modern_gui', 'dither_parameters', value=parameters)
        manager.set('defaults', 'resize_multiplier', value=7)
        manager.save()
        loaded = ConfigManager(path)
        self.assertEqual(loaded.get('modern_gui', 'dither_parameters'), parameters)
        self.assertEqual(loaded.get('defaults', 'resize_multiplier'), 7)
