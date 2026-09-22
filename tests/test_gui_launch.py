"""Routing remains lazy: CLI use must not initialize either GUI."""
import sys
import types
import unittest
from unittest.mock import Mock, patch

import dither_pie


class LaunchTests(unittest.TestCase):
    def test_default_is_modern(self):
        launch = Mock()
        with patch.object(sys, 'argv', ['dither_pie.py']), \
             patch.dict(sys.modules, {'ui.modern.app': types.SimpleNamespace(launch_gui=launch)}):
            dither_pie.main()
        launch.assert_called_once_with()

    def test_explicit_design_routes(self):
        for flag, module in [('--old-design', 'dither_pie_gui'), ('--old_design', 'dither_pie_gui'),
                             ('--new-design', 'ui.modern.app'), ('--new_design', 'ui.modern.app')]:
            launch = Mock()
            with self.subTest(flag=flag), patch.object(sys, 'argv', ['dither_pie.py', flag]), \
                 patch.dict(sys.modules, {module: types.SimpleNamespace(launch_gui=launch)}):
                dither_pie.main()
                launch.assert_called_once_with()

    def test_cli_unchanged(self):
        cli = Mock()
        with patch.object(sys, 'argv', ['dither_pie.py', '--help']), \
             patch.dict(sys.modules, {'dither_cli': types.SimpleNamespace(main=cli)}):
            dither_pie.main()
        cli.assert_called_once_with()

    def test_mixed_design_arguments_rejected(self):
        with patch.object(sys, 'argv', ['dither_pie.py', '--old-design', '--help']):
            with self.assertRaises(SystemExit):
                dither_pie.main()
