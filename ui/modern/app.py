"""Production five-panel GUI using the existing processing and editor contracts."""
from pathlib import Path
from queue import SimpleQueue
import signal
import threading
import tkinter as tk
from tkinter import messagebox

import customtkinter as ctk

from dither_pie_gui import DitheringApp
from dithering_lib import DitherMode, ImageDitherer
from ui.preview_jobs import LatestPreviewJob
from ui.palettes import CustomPaletteCreator
from . import theme as T, winchrome
from .chrome import WindowChrome, RailButton, Logo
from .operations import Operations
from .palette_panel import PalettePanel
from .preferences import Preferences
from .dialogs import ModeGallery, CommandPalette
from .processing import GENERATED
from .viewer import ImageCanvas
from .widgets import (Btn, Card, IconBtn, Label, PaletteCard, RecentFileButton, Segmented, SliderField,
                      StatusBar, SwatchStrip, Toggle, make_entry, make_menu)


class ApplicationStatus(StatusBar):
    def set_status(self, text, spinning=False):
        self.set(text, 'busy' if spinning else 'idle', busy=spinning)


class Viewer(ImageCanvas):
    """Small compatibility surface for the retained production editor workflow."""
    def set_image(self, image, update=True, fit=None):
        super().set_image(image, fit=update if fit is None else fit)

    def fit_to_window(self):
        self.fit()




class ModernApp(PalettePanel, Operations, WindowChrome, winchrome.StableTitlebar, DitheringApp):
    def __init__(self):
        self._closing = False
        self._shutdown_requested = False
        self._image_jobs = LatestPreviewJob()
        self._preview_jobs = LatestPreviewJob()
        self._palette_jobs = LatestPreviewJob()
        self._video_jobs = LatestPreviewJob()
        self._progress = SimpleQueue()
        self._cancel_video = threading.Event()
        self._image_busy = self._video_busy = False
        self._preview_dirty = False
        self._preview_enabled_for_source = False
        self._preview_after = None
        self._palette_search_after = None
        self._rendered_request = None
        self._pixelize_spec = None
        self._view_revision = 0
        self._numbers = []
        self._custom_chrome = self._maximized = self._snapped = False
        self._normal_geometry = self._normal_rect = None
        self._chrome_transition = False
        self._system_maximize_after = None
        self._drag_from = self._manual_drag = None
        super().__init__()
        self.title('Dither Pie')
        self.minsize(1080, 700)
        self.configure(fg_color=T.c('bg'))
        self.bind('<Configure>', self._catch_system_maximize, add='+')
        self.after(140, lambda: self.set_custom_chrome(
            self.config.get('modern_gui', 'custom_titlebar', default=True)))
        self.after(30, self._poll_work)
        self._bind_shortcuts()
        self._refresh_capabilities()

    def _create_sidebar(self):
        T.set_system(self.config.get('modern_gui', 'design_system', default='Teenage Engineering'))
        T.init_fonts()
        self.dither_parameters = self.config.get('modern_gui', 'dither_parameters', default={})
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_columnconfigure(2, weight=1)
        self._build_topbar()
        self.rail = ctk.CTkFrame(self, width=T.RAIL_W, corner_radius=0, fg_color=T.c('surface'))
        self.rail.grid(row=1, column=0, sticky='ns')
        self.rail.grid_propagate(False)
        self.rail.grid_columnconfigure(0, weight=1)
        self.rail_buttons = []
        names = [('Source', 'image'), ('Pixelize', 'grid'), ('Palette', 'palette'),
                 ('Dither', 'sliders'), ('Export', 'arrow_ur')]
        for i, (name, icon) in enumerate(names):
            button = RailButton(self.rail, icon, name, lambda n=i: self.select_panel(n))
            button.grid(row=i, column=0, sticky='ew', padx=6, pady=(6, 0))
            self.rail_buttons.append(button)
        self.sidebar_frame = ctk.CTkFrame(self, width=310, corner_radius=0, fg_color=T.c('bg'))
        self.sidebar_frame.grid(row=1, column=1, sticky='ns')
        self.sidebar_frame.grid_propagate(False)
        self.sidebar_frame.grid_rowconfigure(1, weight=1)
        self.sidebar_frame.grid_columnconfigure(0, weight=1)
        self.panel_title = Label(self.sidebar_frame, 'Source', 'h1')
        self.panel_title.grid(row=0, column=0, sticky='w', padx=18, pady=(18, 10))
        self.panels = []
        for _ in names:
            panel = ctk.CTkScrollableFrame(self.sidebar_frame, fg_color='transparent')
            panel.grid_columnconfigure(0, weight=1)
            self.panels.append(panel)
        self._build_source()
        self._build_pixelize()
        self._build_palette()
        self._build_dither()
        self._build_export()
        self.select_panel(0)

    def _build_topbar(self):
        bar = ctk.CTkFrame(self, height=T.TOPBAR_H, corner_radius=0, fg_color=T.c('surface'))
        bar.grid(row=0, column=0, columnspan=3, sticky='ew')
        bar.grid_propagate(False)
        bar.grid_columnconfigure(1, weight=1)
        bar.grid_rowconfigure(0, weight=1)
        self.topbar = bar
        left = ctk.CTkFrame(bar, fg_color='transparent')
        left.grid(row=0, column=0, padx=16)
        self.logo = Logo(left)
        self.logo.pack(side='left', padx=(0, 10))
        title = Label(left, 'Dither Pie', 'display')
        title.pack(side='left')
        self.file_label = Label(bar, 'Open an image or video', 'mono_s', T.c('dim'))
        self.file_label.grid(row=0, column=1)
        right = ctk.CTkFrame(bar, fg_color='transparent')
        right.grid(row=0, column=2, padx=8)
        IconBtn(right, 'search', lambda: CommandPalette(self), tooltip='Commands (Ctrl+K)').pack(side='left')
        IconBtn(right, 'sun', self.toggle_theme, tooltip='Light / dark').pack(side='left')
        IconBtn(right, 'settings', self.open_settings, tooltip='Preferences').pack(side='left')
        self.win_controls = ctk.CTkFrame(bar, fg_color='transparent')
        self.win_buttons = {}
        for i, (key, icon, command) in enumerate([
            ('min', 'win_min', self.act_minimize), ('max', 'win_max', self.toggle_maximize),
            ('close', 'win_close', self._on_closing)]):
            button = ctk.CTkButton(self.win_controls, text=T.ICON[icon], width=44,
                height=T.TOPBAR_H, corner_radius=0, font=T.icon_font(11), fg_color='transparent',
                hover_color=T.c('err') if key == 'close' else T.c('card_hi'),
                text_color=T.c('text'), command=command)
            button.grid(row=0, column=i)
            self.win_buttons[key] = button
        for widget in (bar, left, title, self.logo, self.file_label):
            self._bind_drag(widget)

    def _card(self, panel, title, icon):
        card = Card(self.panels[panel], title, icon)
        card.grid(row=len(self.panels[panel].winfo_children()), column=0, sticky='ew', pady=(0, 10))
        card.body.grid_columnconfigure(0, weight=1)
        return card.body

    def _button(self, parent, text, command, icon=None):
        button = Btn(parent, text, icon, command=command)
        button.grid(row=len(parent.winfo_children()), column=0, sticky='ew', pady=4)
        return button

    def _number(self, parent, label, low, high, value, step=1, typed_bounds=None, command=None):
        field = SliderField(parent, label, low, high, value, step,
                            typed_bounds=typed_bounds, command=command)
        field.grid(row=len(parent.winfo_children()), column=0, sticky='ew', pady=6)
        self._numbers.append(field)
        return field

    def _build_source(self):
        body = self._card(0, 'Open', 'image')
        self.btn_load_img = self._button(body, 'Open image', self.load_image, 'image')
        self.btn_load_video = self._button(body, 'Open video', self.load_video, 'video')
        self.random_frame_button = self._button(body, 'Random video frame', self.load_random_frame)
        self.source_label = Label(body, 'No file loaded', 'small', wraplength=245)
        self.source_label.grid(row=4, column=0, sticky='w', pady=8)
        self.recent_body = self._card(0, 'Recent files', 'folder')
        self._refresh_recent()

    def _refresh_recent(self):
        for widget in self.recent_body.winfo_children():
            widget.destroy()
        for path in self.config.get('recent_files', default=[])[:10]:
            if isinstance(path, dict):
                path = path.get('path', '')
            if path:
                button = RecentFileButton(self.recent_body, path,
                    lambda p=path: (self.load_video(p) if Path(p).suffix.lower() in
                                    ('.mp4', '.avi', '.mov', '.mkv', '.webm') else self.load_image(p)))
                button.grid(row=len(self.recent_body.winfo_children()), column=0, sticky='ew', pady=3)

    def _build_pixelize(self):
        body = self._card(1, 'Target size', 'crop')
        field = self._number(body, 'Shorter edge (px)', 16, 1024,
            self.config.get('defaults', 'max_size'), 8, (2, None))
        self.max_size_entry = field.entry
        Label(body, 'Even dimensions; nearest-neighbor sampling.', 'tiny', wraplength=245).grid(row=2, column=0)
        self.btn_pixelize = self._button(body, 'Pixelize', self._on_pixelize_regular)
        self.btn_pixelize_neural = self._button(body, 'Neural pixelize', self._on_pixelize_neural)
        self._button(body, 'Use original size', self._use_original)
        body = self._card(1, 'Pixelization editor', 'pencil')
        self.alpha_cutoff_entry = self._number(body, 'Alpha cutoff', 0, 255,
            self.config.get('defaults', 'alpha_cutoff')).entry
        self.btn_pixelize_editor = self._button(body, 'Open editor', self._on_pixelization_editor, 'pencil')
        Label(body, 'Align the source to the grid, convert, then edit with brush, wand, picker and undo/redo.',
              'small', wraplength=245, justify='left').grid(row=4, column=0, pady=8)







    def _build_dither(self):
        body = self._card(3, 'Algorithm', 'sliders')
        modes = [mode.value for mode in DitherMode]
        initial = self.config.get('defaults', 'dither_mode')
        self.dither_mode = tk.StringVar(value=initial if initial in modes else DitherMode.BAYER.value)
        self.dither_dropdown = make_menu(body, modes, self.dither_mode, self._on_dither_mode_changed)
        self.dither_dropdown.grid(row=0, column=0, sticky='ew', pady=8)
        self.dither_settings_button = self._button(body, 'Reset mode parameters', self._reset_parameters)
        self._button(body, 'Browse algorithms', lambda: ModeGallery(self), 'grid_fine')
        self.parameter_body = self._card(3, 'Parameters', 'sliders')
        body = self._card(3, 'Preview', 'eye')
        toggle = Toggle(body, 'Live preview', value=self.config.get('modern_gui', 'live_preview'),
                        command=self._request_preview)
        toggle.grid(row=0, column=0, sticky='ew')
        self.live_var = toggle.var
        self.btn_dither = self._button(body, 'Update preview', self._on_apply_dithering)
        self._build_parameters()

    def _build_parameters(self):
        for child in self.parameter_body.winfo_children():
            if child in self._numbers:
                self._numbers.remove(child)
            child.destroy()
        self.parameter_fields = {}
        mode = self.dither_mode.get()
        info = ImageDitherer.get_mode_parameters(DitherMode(mode)) or {}
        for key, spec in info.items():
            value = self.dither_parameters.get(mode, {}).get(key, spec['default'])
            def change(value, key=key, mode=mode):
                self.dither_parameters.setdefault(mode, {})[key] = value
                self._request_preview()
            if spec['type'] == 'choice':
                Label(self.parameter_body, spec.get('label', key), 'small').grid(
                    row=len(self.parameter_body.winfo_children()), column=0, sticky='w')
                widget = make_menu(self.parameter_body, [str(v) for v in spec['choices']], command=change)
                widget.set(str(value))
                widget.grid(row=len(self.parameter_body.winfo_children()), column=0, sticky='ew', pady=5)
            else:
                widget = SliderField(self.parameter_body, spec.get('label', key), spec['min'], spec['max'],
                    value, spec.get('step', 1 if spec['type'] == 'int' else .05),
                    integer=spec['type'] == 'int', command=change)
                widget.grid(row=len(self.parameter_body.winfo_children()), column=0, sticky='ew', pady=6)
                self._numbers.append(widget)
            self.parameter_fields[key] = widget
            if spec.get('description'):
                Label(self.parameter_body, spec['description'], 'tiny', wraplength=240, justify='left').grid(
                    row=len(self.parameter_body.winfo_children()), column=0, sticky='w')
        if not info:
            Label(self.parameter_body, 'No extra parameters for this mode.', 'small', wraplength=245).grid(row=0, column=0)

    def _on_dither_mode_changed(self, _value):
        self._build_parameters()
        self._update_dither_settings_button_state()
        self._request_preview(activate=True)

    def _reset_parameters(self):
        self.dither_parameters.pop(self.dither_mode.get(), None)
        self._build_parameters()
        self._request_preview()

    def _build_export(self):
        body = self._card(4, 'Output size', 'arrow_ur')
        toggle = Toggle(body, 'Integer upscale', 'Nearest-neighbor; preserves pixel edges.',
                        self.config.get('defaults', 'final_resize_enabled'), command=self._update_resize_preview)
        toggle.grid(row=0, column=0, sticky='ew')
        self.final_resize_var = toggle.var
        self.resize_multiplier_entry = self._number(body, 'Multiplier', 1, 16,
            self.config.get('defaults', 'resize_multiplier', default=2), 1, (1, None),
            lambda _: self._update_resize_preview()).entry
        self.resize_preview_label = Label(body, '', 'small')
        self.resize_preview_label.grid(row=3, column=0, sticky='w')
        self.btn_save = self._button(body, 'Save image', self.save_image, 'save')
        body = self._card(4, 'Video', 'video')
        self.workers = self._number(body, 'Worker processes', 1, 16,
            self.config.get('modern_gui', 'video_workers', default=4), 1, (1, None))
        self.apply_video_button = self._button(body, 'Process and save video', self._apply_to_video_workflow)
        self.cancel_video_button = self._button(body, 'Cancel video processing', self._cancel_video.set)
        self.cancel_video_button.configure(state='disabled')

    def _create_main_area(self):
        self.main_frame = ctk.CTkFrame(self, corner_radius=T.R_CARD, fg_color=T.c('canvas'),
                                     border_width=1, border_color=T.c('border'))
        self.main_frame.grid(row=1, column=2, sticky='nsew', padx=(6, 14), pady=14)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.image_viewer = Viewer(self.main_frame, on_zoom_change=self._zoom_changed)
        self.image_viewer.grid(row=0, column=0, sticky='nsew', padx=1, pady=1)
        self.view_seg = Segmented(self.main_frame, ['Original', 'Pixelized', 'Dithered'],
                                  command=self.show_view, seg_width=95)
        self.view_seg.place(x=14, y=14)
        self.preview_feedback = ctk.CTkFrame(self.main_frame, fg_color=T.c('surface'),
            corner_radius=T.R_CTRL, border_width=1, border_color=T.c('border'))
        self.preview_feedback.place(relx=1, x=-14, y=14, anchor='ne')
        self.preview_feedback_label = Label(self.preview_feedback, 'Open an image or video',
            'small', height=18, wraplength=220, justify='left')
        self.preview_feedback_label.pack(padx=10, pady=7)
        self.preview_activity = ctk.CTkProgressBar(self.preview_feedback, width=220, height=4,
            mode='indeterminate', progress_color=T.accent(), fg_color=T.c('border'))
        controls = ctk.CTkFrame(self.main_frame, fg_color=T.c('surface'))
        controls.place(relx=.5, rely=1, y=-16, anchor='s')
        self.btn_fit = IconBtn(controls, 'fit', self.fit_to_window, tooltip='Fit (F)')
        self.btn_fit.pack(side='left')
        IconBtn(controls, 'zoom_out', lambda: self.image_viewer.zoom_by(1 / 1.25)).pack(side='left')
        self.zoom_label = Label(controls, '100%', 'mono')
        self.zoom_label.pack(side='left', padx=8)
        IconBtn(controls, 'zoom_in', lambda: self.image_viewer.zoom_by(1.25)).pack(side='left')
        Btn(controls, '1:1', command=self.image_viewer.actual_size, width=40).pack(side='left')
        self.btn_toggle = IconBtn(controls, 'layers', self.toggle_compare, tooltip='Before / after (C)')
        self.btn_toggle.pack(side='left')
        self.compare_base = tk.StringVar(value='Pixelized')
        make_menu(controls, ['Original', 'Pixelized'], self.compare_base,
                  lambda _: self._update_comparison(), width=105).pack(side='left', padx=4)
        IconBtn(controls, 'grid_fine', self.toggle_grid, tooltip='Pixel grid').pack(side='left')

    def _create_status_bar(self):
        self.status_bar = ApplicationStatus(self)
        self.status_bar.grid(row=2, column=0, columnspan=3, sticky='ew')

    def set_status(self, text, tone='idle'):
        if hasattr(self, 'status_bar'):
            self.status_bar.set(text, tone, busy=tone == 'busy')
        if hasattr(self, 'preview_feedback_label'):
            self.preview_feedback_label.configure(text=text,
                text_color=T.c('err') if tone == 'err' else T.c('text'))
            self.preview_activity.stop()
            if tone == 'busy':
                self.preview_activity.pack(padx=10, pady=(0, 7))
                self.preview_activity.start()
            else:
                self.preview_activity.pack_forget()

    def select_panel(self, index):
        for i, panel in enumerate(self.panels):
            if i == index:
                panel.grid(row=1, column=0, sticky='nsew', padx=10)
            else:
                panel.grid_remove()
            self.rail_buttons[i].set_active(i == index)
        self.active_panel = index
        self.panel_title.configure(text=['Source', 'Pixelize', 'Palette', 'Dither', 'Export'][index])

    def show_view(self, name, fit=False):
        self._view_revision += 1
        image, state = {'Original': (self.current_image, 'current'),
                        'Pixelized': (self.pixelized_image, 'pixelized'),
                        'Dithered': (self.dithered_image, 'dithered')}[name]
        if image is None:
            self.view_seg.set({'current': 'Original', 'pixelized': 'Pixelized', 'dithered': 'Dithered'}[self.display_state])
            return
        self.display_state = state
        self.view_seg.set(name)
        self.image_viewer.set_image(image, fit=fit)
        self._update_comparison()

    def _update_comparison(self):
        base = self.current_image
        if self.display_state == 'dithered' and self.compare_base.get() == 'Pixelized':
            base = self.pixelized_image or self.current_image
        self.image_viewer.set_compare_image(base)
        if self.display_state == 'current':
            self.image_viewer.set_compare(False)

    def toggle_compare(self):
        if self.dithered_image is not None or self.pixelized_image is not None:
            on = not self.image_viewer.compare
            if self.display_state == 'current':
                self.show_view('Dithered' if self.dithered_image is not None else 'Pixelized')
            self.image_viewer.set_compare(on)

    def toggle_grid(self):
        self.image_viewer.show_pixel_grid = not self.image_viewer.show_pixel_grid
        self.image_viewer.redraw()

    def _zoom_changed(self, zoom):
        if hasattr(self, 'zoom_label'):
            self.zoom_label.configure(text=f'{zoom * 100:.0f}%')

    def _commit_numbers(self):
        # Commit only edits: unchanged fields must not enqueue another preview.
        for field in self._numbers:
            if field.winfo_exists() and field.entry.get() != field._fmt(field.var.get()):
                field._on_type()

    def _bind_shortcuts(self):
        self.bind('<Control-o>', lambda _: self.load_image())
        self.bind('<Control-s>', lambda _: self.save_image())
        self.bind('<Control-comma>', lambda _: self.open_settings())
        self.bind('<Control-k>', lambda _: CommandPalette(self))
        self.bind('<Control-Tab>', lambda _: self.select_panel((self.active_panel + 1) % 5))
        for key, callback in [('f', self.fit_to_window), ('1', self.image_viewer.actual_size), ('c', self.toggle_compare)]:
            self.bind(f'<Key-{key}>', lambda event, action=callback:
                      None if isinstance(event.widget, (tk.Entry, tk.Text, tk.Spinbox)) else action())

    def open_settings(self):
        for child in self.winfo_children():
            if isinstance(child, Preferences):
                child.deiconify()
                child.lift()
                return
        Preferences(self)

    def toggle_theme(self):
        self.set_appearance('Light' if T.is_dark() else 'Dark')

    def set_appearance(self, mode):
        self.config.set('theme', 'appearance_mode', value=mode.lower())
        ctk.set_appearance_mode(mode.lower())
        self._repaint_raw()

    def set_design_system(self, name):
        before = T.token_pairs()
        T.set_system(name)
        self.config.set('modern_gui', 'design_system', value=name)
        replacements = {}
        ambiguous = set()
        for old, new in zip(before, T.token_pairs()):
            for key, value in ((old, new), (T.res(old), T.res(new))):
                if key in replacements and replacements[key] != value:
                    ambiguous.add(key)
                replacements[key] = value
        for key in ambiguous:
            replacements.pop(key, None)
        def walk(widget):
            if getattr(widget, '_fixed_accent', False):
                return
            if hasattr(widget, '_bg_token'):
                old = widget._bg_token
                old = tuple(old) if isinstance(old, list) else old
                if old in replacements:
                    widget._bg_token = replacements[old]
            for option in ('fg_color', 'hover_color', 'text_color', 'border_color',
                           'button_color', 'button_hover_color', 'progress_color'):
                try:
                    old = widget.cget(option)
                    old = tuple(old) if isinstance(old, list) else old
                    if old in replacements:
                        widget.configure(**{option: replacements[old]})
                except (ValueError, TypeError, tk.TclError, AttributeError):
                    pass
            for child in widget.winfo_children():
                walk(child)
        walk(self)
        self._repaint_raw()

    def _repaint_raw(self):
        self.image_viewer.redraw()
        self.logo.draw()
        for card in self.palette_cards.values():
            card.strip.redraw()
        for child in self.winfo_children():
            if isinstance(child, Preferences):
                child.refresh_samples()

    def _on_closing(self):
        if self._closing:
            return
        if self._video_busy:
            self._cancel_video.set()
            self.set_status('Cancelling video before closing…', 'busy')
            self.after(100, self._on_closing)
            return
        self._commit_numbers()
        for key, value in [('design_system', T.system_name()), ('custom_titlebar', self._custom_chrome),
                           ('live_preview', self.live_var.get()), ('video_workers', self.workers.get()),
                           ('dither_parameters', self.dither_parameters)]:
            self.config.set('modern_gui', key, value=value)
        self.config.set('defaults', 'use_gamma', value=self.gamma_var.get())
        super()._on_closing()

    def destroy(self):
        if self._closing:
            return
        self._closing = True
        self._cancel_video.set()
        for jobs in (self._image_jobs, self._preview_jobs, self._video_jobs, self._palette_jobs):
            jobs.close()
        for token in self.tk.call('after', 'info'):
            self.tk.call('after', 'cancel', token)
        super().destroy()


def launch_gui():
    app = ModernApp()
    previous = signal.signal(signal.SIGINT, lambda *_: setattr(app, '_shutdown_requested', True))
    try:
        app.mainloop()
    finally:
        signal.signal(signal.SIGINT, previous)
        app.destroy()
