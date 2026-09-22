"""Production GUI integration tests with real widgets and temporary preferences."""
from pathlib import Path
import tempfile
import time
import shutil
import subprocess
import unittest
from unittest.mock import patch

from PIL import Image

from config_manager import ConfigManager
from dithering_lib import DitherMode
from ui.modern.app import ModernApp, Preferences
from ui.modern.dialogs import ModeGallery, CommandPalette
from ui.modern.processing import DitherRequest
from ui.pixelization_editor import PixelizationEditorDialog
from utils import PaletteManager
from video_processor import VideoProcessor


class ModernGUITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = ConfigManager(str(self.root / 'config.json'))
        self.palette = PaletteManager(str(self.root / 'palette.json'))
        self.palette.add_palette('Test', ['#000000', '#ffffff'])
        with patch('dither_pie_gui.ConfigManager', return_value=self.config), \
             patch('dither_pie_gui.PaletteManager', return_value=self.palette):
            self.app = ModernApp()
        self.addCleanup(self.app.destroy)
        self.errors = []
        self.app.report_callback_exception = lambda *args: self.errors.append(args)
        self.pump(.3)

    def pump(self, duration=.05):
        until = time.monotonic() + duration
        while time.monotonic() < until:
            self.app.update()
            time.sleep(.005)
        self.assertFalse(self.errors, self.errors)

    def finish(self):
        until = time.monotonic() + 30
        jobs = (self.app._image_jobs, self.app._preview_jobs, self.app._video_jobs)
        while any(j._running or j._pending for j in jobs) or self.app._preview_after is not None:
            self.assertLess(time.monotonic(), until, 'Job timed out')
            self.pump()
        self.pump()

    def load(self):
        path = self.root / 'source.png'
        image = Image.new('RGBA', (24, 16), (160, 50, 100, 255))
        image.paste((0, 0, 0, 0), (0, 0, 4, 4))
        image.save(path)
        self.app.load_image(str(path))
        self.finish()
        return image

    def test_real_image_pixelize_dither_and_save(self):
        source = self.load()
        app = self.app
        self.assertEqual(app.current_image.tobytes(), source.tobytes())
        self.assertEqual(app.btn_pixelize_neural.cget('state'), 'disabled')
        field = next(f for f in app._numbers if f.entry is app.max_size_entry)
        field.var.set(8)
        field._sync_entry()
        app._on_pixelize_regular()
        self.finish()
        self.assertEqual(app.pixelized_image.size, (12, 8))
        app.palette_var.set('Test')
        app.gamma_var.set(True)
        app._on_apply_dithering()
        self.finish()
        expected = DitherRequest(app.dither_mode.get(), 16, True, 'Test',
                                 ((0, 0, 0), (255, 255, 255)), {}).render(app.pixelized_image)[0]
        self.assertEqual(app.dithered_image.tobytes(), expected.tobytes())
        self.assertEqual(app.dithered_image.getchannel('A').tobytes(), app.pixelized_image.getchannel('A').tobytes())
        output = self.root / 'output.png'
        app.final_resize_var.set(True)
        with patch('dither_pie_gui.filedialog.asksaveasfilename', return_value=str(output)), \
             patch('dither_pie_gui.messagebox.showinfo'):
            app.save_image()
        with Image.open(output) as saved:
            self.assertEqual(saved.size, (24, 16))
            self.assertEqual(saved.mode, 'RGBA')
        app._use_original()
        self.finish()
        self.assertIsNone(app.pixelized_image)

    def test_all_real_parameter_schemas_and_numeric_arrows(self):
        for mode in DitherMode:
            self.app.dither_mode.set(mode.value)
            self.app._on_dither_mode_changed(mode.value)
            self.pump(.01)
        field = self.app.workers
        field.entry.delete(0, 'end')
        field.entry.insert(0, '32')
        field._nudge(1)
        self.assertEqual(field.get(), 33)

    def test_gallery_and_command_palette_use_real_actions(self):
        gallery = ModeGallery(self.app)
        deadline = time.monotonic() + 30
        while gallery._jobs._running:
            self.assertLess(time.monotonic(), deadline)
            self.pump()
        self.assertEqual(len(gallery._images), len(list(DitherMode)))
        gallery._choose(DitherMode.NONE.value)
        self.assertEqual(self.app.dither_mode.get(), DitherMode.NONE.value)
        commands = CommandPalette(self.app)
        commands.entry.insert(0, 'fit image')
        commands._filter()
        self.assertEqual(len(commands.filtered), 1)
        commands._first()
        self.pump()
        self.assertIsNone(self.app.grab_current())

    def test_production_editor_apply_preserves_transparency(self):
        self.load()
        app = self.app
        visited = []
        def apply_editor():
            dialog = next(w for w in app.winfo_children() if isinstance(w, PixelizationEditorDialog))
            visited.append(dialog)
            dialog.grid_w, dialog.grid_h = 2, 2
            dialog.canvas.set_grid(2, 2)
            dialog._conversion_done([[(255, 0, 0), None], [None, (0, 0, 0)]], None)
            dialog._on_apply()
        app.after(500, apply_editor)
        app._on_pixelization_editor()
        self.finish()
        self.assertEqual(len(visited), 1)
        self.assertEqual(app.pixelized_image.mode, 'RGBA')
        self.assertEqual(app.pixelized_image.getpixel((1, 0))[3], 0)
        self.assertEqual(app.pixelization_cache['method'], 'editor')

    def test_editor_conversion_runs_off_ui_and_closes_safely(self):
        from threading import get_ident
        from ui.pixelization_editor import sample_editor_grid
        source = self.load()
        dialog = PixelizationEditorDialog(self.app, source, config=self.config, alpha_cutoff=128)
        self.pump(.2)
        owner = get_ident()
        calls = []
        original_transform = dialog.canvas._get_image_transform
        def read_transform():
            self.assertEqual(get_ident(), owner)
            return original_transform()
        def sample(*args):
            calls.append(get_ident())
            time.sleep(.1)
            return sample_editor_grid(*args)
        with patch.object(dialog.canvas, '_get_image_transform', read_transform), \
             patch('ui.pixelization_editor.sample_editor_grid', sample):
            dialog._start_conversion()
            until = time.monotonic() + 5
            while dialog._conversion_jobs._running:
                self.assertLess(time.monotonic(), until)
                self.pump()
        self.assertTrue(calls)
        self.assertNotEqual(calls[0], owner)
        self.assertTrue(dialog.canvas.get_pixel_data())
        dialog._start_conversion()
        dialog._on_cancel()
        self.pump(.2)
        self.assertIsNone(self.app.grab_current())

    def test_preferences_modal_and_persistence(self):
        app = self.app
        self.assertEqual(self.config.get('theme', 'appearance_mode'), 'light')
        app.open_settings()
        self.pump()
        dialog = next(w for w in app.winfo_children() if isinstance(w, Preferences))
        app.set_appearance('Dark')
        app.set_design_system('Swiss Grid')
        self.pump()
        self.assertTrue(dialog.winfo_viewable())
        self.assertIs(app.grab_current(), dialog)
        dialog.close()
        app.open_settings()
        self.pump()
        app.grab_current().close()
        app._on_closing()
        loaded = ConfigManager(str(self.root / 'config.json'))
        self.assertEqual(loaded.get('theme', 'appearance_mode'), 'dark')
        self.assertEqual(loaded.get('modern_gui', 'design_system'), 'Swiss Grid')

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
    def test_real_video_load_preview_and_export(self):
        source = self.root / 'source.mp4'
        output = self.root / 'result.mp4'
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-f', 'lavfi',
            '-i', 'testsrc2=size=32x24:rate=24000/1001:duration=0.25',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(source)],
            capture_output=True, check=True, timeout=30)
        app = self.app
        app.load_video(str(source))
        self.finish()
        self.assertTrue(app.is_video)
        self.assertEqual(app.btn_pixelize_editor.cget('state'), 'disabled')
        self.assertEqual(app.current_image.size, (32, 24))
        app.palette_var.set('Test')
        app._on_apply_dithering()
        self.finish()
        app.final_resize_var.set(True)
        app.workers.var.set(1)
        app.workers._sync_entry()
        with patch('ui.modern.operations.filedialog.asksaveasfilename', return_value=str(output)):
            app._apply_to_video_workflow()
        self.finish()
        self.assertTrue(output.exists())
        info = VideoProcessor().get_video_info(str(output))
        self.assertEqual((info['width'], info['height']), (64, 48))
        self.assertEqual(info['fps_fraction'], '24000/1001')
        self.assertFalse(app.status_bar._progress_shown)

    def test_late_preview_cannot_replace_new_source(self):
        from threading import Event
        started, release = Event(), Event()
        self.addCleanup(release.set)
        self.load()
        app = self.app
        app.palette_var.set('Test')
        original = DitherRequest.render
        def slow(request, source):
            started.set()
            release.wait(5)
            return original(request, source)
        with patch.object(DitherRequest, 'render', slow):
            app._on_apply_dithering()
            self.assertTrue(started.wait(5))
            app._source_loaded(Image.new('RGB', (10, 10)), self.root / 'new.png')
            release.set()
            self.finish()
        self.assertIsNone(app.dithered_image)
        self.assertEqual(app.current_image.size, (10, 10))


if __name__ == '__main__':
    unittest.main()
