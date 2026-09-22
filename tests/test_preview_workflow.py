"""Run the real palette workflow with fake widgets and a controlled slow worker."""

from contextlib import ExitStack
from threading import Event, get_ident
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from dither_pie_gui import DitheringApp


class PreviewWorkflowTests(unittest.TestCase):
    def run_workflow(self, cancel=False):
        owner = get_ident()
        variables, buttons, checkboxes, scheduled = [], {}, [], {}
        started, release = Event(), Event()
        self.addCleanup(release.set)

        class Variable:
            def __init__(self, value=None):
                self.value, self.callbacks = value, []
                variables.append(self)

            def get(self):
                if get_ident() != owner:
                    raise AssertionError('Tk variable read from worker')
                return self.value

            def set(self, value):
                self.value = value
                for callback in self.callbacks:
                    callback()

            def trace_add(self, mode, callback):
                self.callbacks.append(callback)
                return 'trace'

        dialog = Mock()
        alive = [True]
        dialog.geometry.return_value = '400x600+0+0'
        dialog.winfo_exists.side_effect = lambda: alive[0]
        dialog.destroy.side_effect = lambda: alive.__setitem__(0, False)

        def schedule(delay, callback):
            token = object()
            scheduled[token] = callback
            return token

        dialog.after.side_effect = schedule
        dialog.after_cancel.side_effect = lambda token: scheduled.pop(token, None)

        def button(*args, **kwargs):
            widget = Mock()
            widget.invoke.side_effect = kwargs['command']
            buttons[kwargs['text']] = widget
            return widget

        def checkbox(*args, **kwargs):
            checkboxes.append(kwargs)
            return Mock()

        old_image = Image.new('RGB', (4, 4), 'red')
        viewer = Mock(zoom_factor=1, offset_x=0, offset_y=0)
        shown = []
        viewer.update_view.side_effect = lambda: shown.append(viewer.original_image)
        app = SimpleNamespace(
            colors_entry=Mock(get=Mock(return_value='2')),
            pixelized_image=None, current_image=Image.new('RGB', (4, 4)),
            dithered_image=old_image, display_state='dithered', config=Mock(),
            _disable_controls=Mock(), _enable_controls=Mock(), _update_resize_preview=Mock(),
            image_viewer=viewer, status_bar=Mock(), palette_preview_zoom_state=None,
            palette_mgr=SimpleNamespace(palettes=[]), dither_parameters={},
            dither_mode=SimpleNamespace(get=lambda: 'bayer'),
        )
        app.config.get.side_effect = lambda *keys, **kwargs: kwargs.get('default')
        calls = []

        class Ditherer:
            def __init__(self, **settings):
                self.settings = settings

            def apply_dithering(self, source):
                calls.append(self.settings)
                if len(calls) == 1:
                    started.set()
                    release.wait(3)
                color = self.settings['palette'][0]
                return Image.new('RGB', source.size, color)

        def interact(window):
            self.assertTrue(started.wait(3))
            if cancel:
                buttons['Cancel'].invoke()
                release.set()
                return
            selection = next(v for v in variables if v.value == 'Median Cut')
            selection.set('Uniform')
            checkboxes[0]['variable'].set(True)
            checkboxes[0]['command']()
            buttons['Apply Selected'].invoke()
            self.assertTrue(alive[0], 'Apply closed before the selected preview finished')
            self.assertIs(app.dithered_image, old_image)
            release.set()
            deadline = time.monotonic() + 3
            while alive[0] and time.monotonic() < deadline:
                for token, callback in list(scheduled.items()):
                    scheduled.pop(token, None)
                    callback()
                time.sleep(0.005)
            self.assertFalse(alive[0], 'Apply did not finish')

        app.wait_window = interact
        with ExitStack() as stack:
            for name in ('CTkLabel', 'CTkFrame', 'CTkRadioButton'):
                stack.enter_context(patch('dither_pie_gui.ctk.' + name))
            scroll = stack.enter_context(patch('dither_pie_gui.ctk.CTkScrollableFrame'))
            scroll.return_value.winfo_children.return_value = []
            stack.enter_context(patch('dither_pie_gui.ctk.CTkToplevel', return_value=dialog))
            stack.enter_context(patch('dither_pie_gui.ctk.CTkButton', side_effect=button))
            stack.enter_context(patch('dither_pie_gui.ctk.CTkCheckBox', side_effect=checkbox))
            stack.enter_context(patch('dither_pie_gui.tk.BooleanVar', Variable))
            stack.enter_context(patch('dither_pie_gui.tk.StringVar', Variable))
            stack.enter_context(patch('gui_components.PalettePreview'))
            stack.enter_context(patch('dither_pie_gui.ImageDitherer', Ditherer))
            for name, color in [('reduce_colors', (10, 10, 10)), ('generate_kmeans_palette', (20, 20, 20)),
                                ('generate_uniform_palette', (30, 30, 30))]:
                stack.enter_context(patch('dither_pie_gui.ColorReducer.' + name, return_value=[color]))
            DitheringApp._show_palette_dialog_and_apply(app)
        return app, calls, shown, old_image

    def test_apply_waits_for_latest_palette_and_gamma(self):
        app, calls, shown, old = self.run_workflow()
        self.assertEqual(app.dithered_image.getpixel((0, 0)), (30, 30, 30))
        self.assertTrue(app.last_gamma)
        self.assertTrue(calls[-1]['use_gamma'])
        self.assertTrue(all(image.getpixel((0, 0)) == (30, 30, 30) for image in shown))

    def test_cancel_keeps_previous_image(self):
        app, calls, shown, old = self.run_workflow(cancel=True)
        self.assertIs(app.dithered_image, old)
        self.assertIs(shown[-1], old)

    def test_gui_rejects_size_one_before_starting_a_worker(self):
        app = SimpleNamespace(current_image=Image.new('RGB', (4, 4)),
                              max_size_entry=Mock(get=Mock(return_value='1')))
        for method in (DitheringApp._on_pixelize_regular, DitheringApp._on_pixelize_neural):
            with self.subTest(method=method.__name__), \
                 patch('dither_pie_gui.messagebox.showerror') as error, \
                 patch('dither_pie_gui.threading.Thread') as worker:
                method(app)
                error.assert_called_once()
                worker.assert_not_called()
