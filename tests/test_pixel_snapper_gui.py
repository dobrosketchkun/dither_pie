"""Real-widget coverage for standalone snapping in both GUI designs."""
import gc
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from config_manager import ConfigManager
from dither_pie_gui import DitheringApp
from ui.modern.app import ModernApp
from ui.pixel_snapper import PixelSnapperDialog
from ui.modern.pixel_snapper import ModernPixelSnapperDialog
from ui.snapper_color import SnapColorDialog
from utils import PaletteManager


class SnapperGUITests(unittest.TestCase):
    def test_add_delete_and_image_picker_in_both_designs(self):
        def scenario(app, modern, errors):
            app.config.set('pixel_snapper', 'num_colors', value=1)
            dialog_type = ModernPixelSnapperDialog if modern else PixelSnapperDialog
            dialog = dialog_type(app, app.current_image, config=app.config)
            self.finish_dialog(app, dialog)
            self.assertEqual(len(dialog.palette_colors), 1)
            with patch('ui.pixel_snapper.SnapColorDialog', return_value=SimpleNamespace(selected_color=(20, 60, 90))), \
                    patch.object(dialog, 'wait_window'), patch('ui.pixel_snapper.snap_pixels') as regenerate:
                dialog._add_color()
                self.assertIsNone(dialog.preview_result)
                self.assertFalse(dialog._jobs._running)
                dialog._preview()
                self.finish_dialog(app, dialog)
                regenerate.assert_not_called()
            self.assertEqual(len(dialog.palette_colors), 2)
            self.assertIn((20, 60, 90), dialog.preview_result.image.getdata())
            dialog._delete_color()
            dialog._preview()
            self.finish_dialog(app, dialog)
            self.assertEqual(len(dialog.palette_colors), 1)
            dialog._delete_color()
            self.assertEqual(len(dialog.palette_colors), 1)

            picker = SnapColorDialog(dialog, (64, 0, 64))
            self.assertEqual(picker.get_rgb(), (64, 0, 64))
            for view in (dialog.source_view, dialog.result_view):
                dialog._start_color_pick(picker)
                view.zoom(SimpleNamespace(x=90, y=100, delta=120, state=0))
                x, y, w, h = view.geometry_of_image()
                # Sample the first source cell after zoom, using the same
                # inverse transform as painting; no screen/composite sampling.
                event = SimpleNamespace(x=x + w * .0625, y=y + h * .1)
                expected = view.color_at(event.x, event.y)
                view.start_pan(event)
                self.assertEqual(picker.get_rgb(), expected)
                self.assertIsNone(dialog._color_picker)
                self.assertIsNone(view.pick_callback)
            dialog._start_color_pick(picker)
            dialog._escape()
            self.assertIsNone(dialog._color_picker)
            picker.destroy()
            dialog.grab_set()
            # Outside-image clicks do not select the canvas background.
            self.assertIsNone(dialog.source_view.color_at(-10000, -10000))
            dialog.algorithm_var.set('Octree')
            self.assertFalse(dialog._jobs._running)
            dialog._preview()
            self.finish_dialog(app, dialog)
            self.assertFalse(dialog._palette_custom)
            dialog.close()
        for modern in (False, True):
            with self.subTest(modern=modern):
                self.run_design(modern, scenario)

    def test_preview_camera_is_shared_and_survives_resolution_changes(self):
        def scenario(app, modern, errors):
            dialog_type = ModernPixelSnapperDialog if modern else PixelSnapperDialog
            dialog = dialog_type(app, app.current_image, config=app.config)
            self.finish_dialog(app, dialog)
            source, result = dialog.source_view, dialog.result_view
            self.assertIs(source.camera, result.camera)
            self.assertEqual(source.geometry_of_image()[2:], result.geometry_of_image()[2:])
            source.zoom(SimpleNamespace(x=80, y=90, delta=120, num=0, state=0))
            result.start_pan(SimpleNamespace(x=100, y=110))
            result.pan(SimpleNamespace(x=132, y=94))
            camera = dialog.camera
            before = (camera.scale, camera.offset_x, camera.offset_y, camera.auto_fit)
            dialog._preview()
            self.finish_dialog(app, dialog)
            self.assertEqual(before, (camera.scale, camera.offset_x, camera.offset_y, camera.auto_fit))
            dialog.size_var.set('2')
            dialog._preview()
            self.finish_dialog(app, dialog)
            self.assertEqual(dialog.preview_result.image.size, (16, 10))
            self.assertEqual(before, (camera.scale, camera.offset_x, camera.offset_y, camera.auto_fit))
            self.assertEqual(source.geometry_of_image()[2:], result.geometry_of_image()[2:])
            if modern:
                self.assertEqual(len(app.rail_buttons), 5)
                self.assertFalse(hasattr(app, 'snapper_rail'))
                app.select_panel(1)
                self.assertTrue(app.btn_pixel_snapper.winfo_ismapped())
            dialog.close()
        for modern in (False, True):
            with self.subTest(modern=modern):
                self.run_design(modern, scenario)

    def test_swatches_edit_colors_and_reduction_changes_reset_them(self):
        def scenario(app, modern, errors):
            dialog_type = ModernPixelSnapperDialog if modern else PixelSnapperDialog
            dialog = dialog_type(app, app.current_image, config=app.config)
            self.finish_dialog(app, dialog)
            generated = tuple(dialog.palette_colors)
            before = dialog.result_view.original_image.tobytes()
            self.assertEqual(len(dialog.palette_frame.winfo_children()), len(generated))
            dialog._set_palette_color(0, (1, 2, 3))
            self.assertEqual(dialog.result_view.original_image.tobytes(), before)
            self.assertFalse(dialog._jobs._running)
            self.assertEqual(dialog.apply_button.cget('state'), 'disabled')
            self.assertEqual(dialog.delete_color_button.cget('state'), 'normal')
            dialog._preview()
            self.finish_dialog(app, dialog)
            self.assertEqual(dialog.palette_colors[0], (1, 2, 3))
            self.assertIn((1, 2, 3), dialog.preview_result.image.getdata())
            self.assertIsNotNone(dialog._edited_palette)
            dialog.algorithm_var.set('Octree')
            self.assertIsNone(dialog._edited_palette)
            self.assertFalse(dialog._jobs._running)
            dialog._preview()
            self.finish_dialog(app, dialog)
            self.assertEqual(tuple(dialog.palette_colors), generated)
            self.assertNotIn((1, 2, 3), dialog.preview_result.image.getdata())
            dialog._set_palette_color(0, (1, 2, 3))
            dialog._preview()
            self.finish_dialog(app, dialog)
            dialog._reset_palette()
            self.assertFalse(dialog._jobs._running)
            self.assertEqual(dialog.palette_colors, list(generated))
            dialog._preview()
            self.finish_dialog(app, dialog)
            self.assertEqual(tuple(dialog.palette_colors), generated)
            dialog.close()
        for modern in (False, True):
            with self.subTest(modern=modern):
                self.run_design(modern, scenario)

    def run_design(self, modern, scenario):
        gc.collect()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = ConfigManager(str(root / 'config.json'))
            config.set('pixel_snapper', 'pixel_size', value=4)
            palette = PaletteManager(str(root / 'palette.json'))
            with patch('dither_pie_gui.ConfigManager', return_value=config), \
                    patch('dither_pie_gui.PaletteManager', return_value=palette):
                app = ModernApp() if modern else DitheringApp()
            try:
                errors = []
                app.report_callback_exception = lambda *args: errors.append(args)
                source = Image.fromarray(np.tile(np.array([[[20, 60, 90], [200, 150, 30]]], dtype=np.uint8), (5, 4, 1)))
                source = source.resize((32, 20), Image.Resampling.NEAREST)
                app.current_image = source
                app.original_size = source.size
                app.is_video = False
                app.source_has_alpha = False
                app.update()
                scenario(app, modern, errors)
                self.assertFalse(errors, errors)
            finally:
                # CustomTkinter schedules global DPI/appearance polling; cancel
                # it before replacing a legacy root with the next test root.
                for token in app.tk.call('after', 'info'):
                    app.tk.call('after', 'cancel', token)
                app.destroy()

    def finish_dialog(self, app, dialog):
        deadline = time.monotonic() + 20
        while dialog._jobs._running or dialog._jobs._pending:
            self.assertLess(time.monotonic(), deadline)
            app.update()
            time.sleep(.005)
        app.update()

    def test_preview_changes_cannot_apply_stale_result_and_cancel_is_inert(self):
        def scenario(app, modern, errors):
            dialog_type = ModernPixelSnapperDialog if modern else PixelSnapperDialog
            dialog = dialog_type(app, app.current_image, config=app.config)
            self.finish_dialog(app, dialog)
            self.assertIsNotNone(dialog.preview_result)
            self.assertEqual(dialog.preview_result.image.size, (8, 5))
            self.assertEqual(dialog.apply_button.cget('state'), 'normal')
            dialog.size_var.set('2')
            self.assertEqual(dialog.apply_button.cget('state'), 'disabled')
            dialog._apply()
            self.assertIsNone(dialog.result_image)
            dialog._preview()
            self.finish_dialog(app, dialog)
            self.assertEqual(dialog.preview_result.image.size, (16, 10))
            dialog.close()
            self.assertIsNone(app.pixelized_image)
            self.assertEqual(app.config.get('pixel_snapper', 'pixel_size'), 4)
        for modern in (False, True):
            with self.subTest(modern=modern):
                self.run_design(modern, scenario)

    def test_direct_operation_applies_and_clears_previous_dithering(self):
        def scenario(app, modern, errors):
            old_source = app.current_image.tobytes()
            app.dithered_image = Image.new('RGB', (3, 3))
            deadline = time.monotonic() + 20
            def apply_when_ready():
                dialogs = [child for child in app.winfo_children()
                           if isinstance(child, (PixelSnapperDialog, ModernPixelSnapperDialog))]
                if dialogs and dialogs[0].preview_result is not None:
                    dialogs[0]._apply()
                elif time.monotonic() < deadline:
                    app.after(20, apply_when_ready)
                elif dialogs:
                    errors.append(('Preview timeout', dialogs[0].status.cget('text')))
                    dialogs[0].close()
            app.after(20, apply_when_ready)
            app._on_pixel_snapper()
            self.assertEqual(app.pixelized_image.size, (8, 5))
            self.assertIsNone(app.dithered_image)
            self.assertEqual(app.pixelization_cache['method'], 'snapper')
            self.assertEqual(app.current_image.tobytes(), old_source)
            self.assertEqual(app.display_state, 'pixelized')
            if modern:
                self.assertIsNone(app._pixelize_spec)
            editor_path = ('ui.modern.editor.ModernPixelizationEditor' if modern
                           else 'dither_pie_gui.PixelizationEditorDialog')
            with patch(editor_path, return_value=SimpleNamespace(result_image=None)) as editor, \
                    patch.object(app, 'wait_window'):
                app._on_pixelization_editor()
            self.assertIs(editor.call_args.args[1], app.pixelized_image)
        for modern in (False, True):
            with self.subTest(modern=modern):
                self.run_design(modern, scenario)

    def test_close_discards_running_work_and_invalid_input_disables_apply(self):
        def scenario(app, modern, errors):
            dialog_type = ModernPixelSnapperDialog if modern else PixelSnapperDialog
            dialog = dialog_type(app, app.current_image, config=app.config)
            dialog.count_var.set('0')
            dialog._preview()
            self.assertEqual(dialog.apply_button.cget('state'), 'disabled')
            self.assertIn('1 to 256', dialog.status.cget('text'))
            dialog.count_var.set('16')
            dialog._preview()
            dialog.close()
            self.assertTrue(dialog._jobs._closed)
            end = time.monotonic() + .2
            while time.monotonic() < end:
                app.update()
                time.sleep(.01)
        for modern in (False, True):
            with self.subTest(modern=modern):
                self.run_design(modern, scenario)
