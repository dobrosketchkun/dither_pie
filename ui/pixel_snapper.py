"""Standalone Pixel Snapper dialog shared by both desktop designs."""
import tkinter as tk
import customtkinter as ctk

from pixel_snapper import SnapSettings, snap_pixels, apply_snap_palette
from .snapper_color import SnapColorDialog
from .snapper_preview import PreviewCamera, SyncedPreview
from .preview_jobs import LatestPreviewJob


ALGORITHMS = {'Median Cut': 'median_cut', 'K-means': 'kmeans', 'Octree': 'octree'}


class SnapperContents:
    def _initialize_snapper(self, source, config, video_path=None):
        self.source_image = source.copy()
        self.config_mgr = config
        self.result_image = self.result = self.preview_result = None
        self.video_path = video_path
        self.result_plan = self._preview_video_plan = self._base_video_plan = None
        self._jobs = LatestPreviewJob()
        self._poll_id = None
        self._snap_closed = False
        self._preview_settings = None
        self.palette_colors = []
        self._edited_palette = None
        self._palette_custom = False
        self._base_preview = None
        self._selected_color = 0
        self._color_picker = None
        preferences = config.get('pixel_snapper', default={}) if config else {}
        preferences = dict(preferences) if isinstance(preferences, dict) else {}
        auto_colors = preferences.pop('auto_colors', True)
        try:
            settings = SnapSettings(**preferences)
        except (TypeError, ValueError):
            settings = SnapSettings()
        self.count_var = tk.StringVar(self, str(settings.num_colors or 16))
        self.auto_colors_var = tk.BooleanVar(self, auto_colors)
        name = next(name for name, value in ALGORITHMS.items() if value == settings.quantizer)
        self.algorithm_var = tk.StringVar(self, name)
        self.auto_var = tk.BooleanVar(self, settings.pixel_size is None)
        self.size_var = tk.StringVar(self, str(settings.pixel_size or 4))
        self._build_snapper()
        for variable in (self.auto_var, self.size_var):
            variable.trace_add('write', self._settings_changed)
        self.count_var.trace_add('write', self._palette_settings_changed)
        self.auto_colors_var.trace_add('write', self._palette_settings_changed)
        self.algorithm_var.trace_add('write', self._reduction_changed)
        self.protocol('WM_DELETE_WINDOW', self.close)
        self.bind('<Escape>', self._escape)
        self._settings_changed()
        self._poll_id = self.after(50, self._poll_snapper)
        self._preview()

    def _build_snapper(self):
        self.body.grid_columnconfigure(0, weight=0)
        self.body.grid_columnconfigure(1, weight=1)
        self.body.grid_rowconfigure(0, weight=1)
        settings = ctk.CTkFrame(self.body, width=220)
        settings.grid(row=0, column=0, sticky='ns', padx=(0, 12))
        settings.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(settings, text='Pixel grid', font=('', 14, 'bold')).grid(
            row=0, column=0, sticky='w', padx=12, pady=12)
        self.auto_check = ctk.CTkCheckBox(settings, text='Detect pixel size automatically', variable=self.auto_var)
        self.auto_check.grid(row=1, column=0, sticky='w', padx=12, pady=6)
        ctk.CTkLabel(settings, text='Source cell size (px)').grid(row=2, column=0, sticky='w', padx=12)
        self.size_entry = ctk.CTkEntry(settings, textvariable=self.size_var)
        self.size_entry.grid(row=3, column=0, sticky='ew', padx=12, pady=(0, 12))
        self.auto_colors_check = ctk.CTkCheckBox(settings, text='Auto color count', variable=self.auto_colors_var)
        self.auto_colors_check.grid(row=4, column=0, sticky='w', padx=12, pady=(0, 8))
        ctk.CTkLabel(settings, text='Manual colors (1–256)').grid(row=5, column=0, sticky='w', padx=12)
        self.count_entry = ctk.CTkEntry(settings, textvariable=self.count_var)
        self.count_entry.grid(row=6, column=0, sticky='ew', padx=12)
        ctk.CTkLabel(settings, text='Color reduction').grid(row=7, column=0, sticky='w', padx=12, pady=(12, 0))
        self.algorithm_menu = ctk.CTkOptionMenu(settings, values=list(ALGORITHMS), variable=self.algorithm_var)
        self.algorithm_menu.grid(row=8, column=0, sticky='ew', padx=12)
        self.preview_button = ctk.CTkButton(settings, text='Update preview', command=self._preview)
        self.preview_button.grid(row=9, column=0, sticky='ew', padx=12, pady=16)
        palette_head = ctk.CTkFrame(settings, fg_color='transparent')
        palette_head.grid(row=10, column=0, sticky='ew', padx=12)
        self.palette_label = ctk.CTkLabel(palette_head, text='Palette')
        self.palette_label.pack(side='left')
        self.reset_palette_button = ctk.CTkButton(palette_head, text='Reset', width=55,
                                                 command=self._reset_palette)
        self.reset_palette_button.pack(side='right')
        self.delete_color_button = ctk.CTkButton(palette_head, text='−', width=28, command=self._delete_color)
        self.delete_color_button.pack(side='right', padx=(2, 4))
        self.add_color_button = ctk.CTkButton(palette_head, text='+', width=28, command=self._add_color)
        self.add_color_button.pack(side='right', padx=2)
        settings.grid_rowconfigure(11, weight=1)
        self.palette_frame = ctk.CTkScrollableFrame(settings, width=190, height=150)
        self.palette_frame.grid(row=11, column=0, sticky='nsew', padx=12, pady=(6, 12))
        views = ctk.CTkFrame(self.body, fg_color='transparent')
        views.grid(row=0, column=1, sticky='nsew')
        views.grid_columnconfigure((0, 1), weight=1, uniform='preview')
        views.grid_rowconfigure(1, weight=1)
        self.camera = PreviewCamera(self.source_image.size)
        self.source_view = SyncedPreview(views, self.camera, bg='#363636', highlightthickness=0)
        self.result_view = SyncedPreview(views, self.camera, bg='#363636', highlightthickness=0)
        for column, (label, view) in enumerate((('Source', self.source_view), ('Snapped preview', self.result_view))):
            ctk.CTkLabel(views, text=label).grid(row=0, column=column, sticky='w', padx=6)
            view.grid(row=1, column=column, sticky='nsew', padx=6)
        self.fit_button = ctk.CTkButton(views, text='Fit view', width=70, height=24, command=self.camera.fit)
        self.fit_button.grid(row=0, column=1, sticky='e', padx=6)
        self.source_view.set_image(self.source_image)
        self.status = ctk.CTkLabel(self.footer, text='', anchor='w', justify='left', wraplength=650)
        self.status.pack(side='left', fill='x', expand=True)
        self.apply_button = ctk.CTkButton(self.footer, text='Apply', width=90, state='disabled', command=self._apply)
        self.apply_button.pack(side='right', padx=(8, 0))
        self.cancel_button = ctk.CTkButton(self.footer, text='Cancel', width=90, command=self.close)
        self.cancel_button.pack(side='right')

    def _read_settings(self):
        try:
            count = None if self.auto_colors_var.get() else int(self.count_var.get())
            size = None if self.auto_var.get() else float(self.size_var.get())
        except ValueError:
            raise ValueError('Enter an integer color count and a numeric pixel size.') from None
        return SnapSettings(count, ALGORITHMS[self.algorithm_var.get()], size)

    def _settings_changed(self, *_):
        self._jobs.invalidate()
        self.preview_result = None
        self._preview_settings = None
        self.apply_button.configure(state='disabled')
        self.add_color_button.configure(state='normal' if self._base_preview and len(self.palette_colors) < 256 else 'disabled')
        self.delete_color_button.configure(state='normal' if len(self.palette_colors) > 1 else 'disabled')
        self.size_entry.configure(state='disabled' if self.auto_var.get() else 'normal')
        self.count_entry.configure(state='disabled' if self.auto_colors_var.get() else 'normal')
        self.status.configure(text='Settings changed. Update preview to apply.')

    def _palette_settings_changed(self, *_):
        self.auto_colors_check.configure(text='Auto color count')
        self._edited_palette = None
        self._palette_custom = False
        self._base_preview = None
        self.palette_colors = []
        self._show_palette()
        self._settings_changed()

    def _reduction_changed(self, *_):
        self._palette_settings_changed()

    def _show_palette(self):
        suggested = (self._base_video_plan is not None and self._base_video_plan.preserve_colors
                     and self._edited_palette is None)
        self.palette_label.configure(text=f'{"Suggested" if suggested else "Palette"} ({len(self.palette_colors)})')
        self._selected_color = max(0, min(self._selected_color, len(self.palette_colors) - 1))
        for child in self.palette_frame.winfo_children():
            child.destroy()
        for index, color in enumerate(self.palette_colors):
            hex_color = '#%02x%02x%02x' % tuple(color)
            button = ctk.CTkButton(self.palette_frame, text='', width=28, height=28,
                                  fg_color=hex_color, hover_color=hex_color,
                                  border_width=2 if index == self._selected_color else 1,
                                  border_color=('#222222', '#eeeeee') if index == self._selected_color else ('#888888', '#777777'),
                                  corner_radius=3, command=lambda i=index: self._edit_color(i))
            button.grid(row=index // 6, column=index % 6, padx=2, pady=2)

    def _edit_color(self, index):
        self._selected_color = index
        self._show_palette()
        picker = SnapColorDialog(self, self.palette_colors[index], can_delete=len(self.palette_colors) > 1)
        self.wait_window(picker)
        if not self.winfo_exists():
            return
        self.grab_set()
        if picker.action == 'delete':
            self._delete_color()
        elif picker.selected_color is not None:
            self._set_palette_color(index, picker.selected_color)

    def _set_palette_color(self, index, color):
        self.palette_colors[index] = tuple(color)
        self._submit_palette()

    def _add_color(self):
        if not self._base_preview or len(self.palette_colors) >= 256:
            return
        picker = SnapColorDialog(self, (255, 255, 255))
        self.wait_window(picker)
        if not self.winfo_exists():
            return
        self.grab_set()
        if picker.selected_color is not None:
            self.palette_colors.append(picker.selected_color)
            self._selected_color = len(self.palette_colors) - 1
            self._palette_custom = True
            self._submit_palette()

    def _delete_color(self):
        if len(self.palette_colors) <= 1:
            return
        self.palette_colors.pop(self._selected_color)
        self._palette_custom = True
        self._submit_palette()

    def _submit_palette(self):
        self._edited_palette = tuple(self.palette_colors)
        self._show_palette()
        self._settings_changed()

    def _reset_palette(self):
        self._edited_palette = None
        self._palette_custom = False
        cached = self._base_preview
        self.palette_colors = list(cached[1].palette) if cached else []
        self._show_palette()
        self._settings_changed()

    def _preview(self):
        self._jobs.invalidate()
        self.preview_result = None
        self.apply_button.configure(state='disabled')
        try:
            settings = self._read_settings()
        except ValueError as error:
            self.status.configure(text=str(error))
            return
        self.status.configure(text='Analyzing clip grid and shared palette…' if self.video_path
                              else 'Recovering grid and reducing colors…')
        palette = self._edited_palette
        custom = self._palette_custom
        cached = self._base_preview
        cached_plan = self._base_video_plan
        def compute():
            if self.video_path:
                from video_snapper import analyze_video_snap
                plan = (cached_plan if cached and cached[0] == settings and cached_plan is not None
                        else analyze_video_snap(self.video_path, settings))
                base = plan.preview(self.source_image)
                edited = plan.with_palette(palette, custom) if palette is not None else plan
                return base, edited.preview(self.source_image), plan, edited
            base = cached[1] if cached and cached[0] == settings else snap_pixels(self.source_image, settings)
            result = base if palette is None else apply_snap_palette(self.source_image, base, palette, custom=custom)
            return base, result, None, None
        def done(value):
            base, result, plan, edited = value
            self._base_video_plan, self._preview_video_plan = plan, edited
            self._base_preview = (settings, base)
            self.preview_result = result
            self._preview_settings = settings
            self.result_view.set_image(result.image)
            self.palette_colors = list(result.palette)
            self._show_palette()
            if self.auto_colors_var.get():
                if plan and plan.preserve_colors:
                    color_text = 'Auto colors (original)' if edited.preserve_colors else 'Auto colors (edited)'
                else:
                    color_text = f'Auto color count ({len(base.palette)})'
                self.auto_colors_check.configure(text=color_text)
            text = f'{result.image.width} × {result.image.height} pixels · source cell ≈ {result.pixel_size:g}px'
            if result.warning:
                text += '\n' + result.warning
            if result.color_note:
                text += '\n' + result.color_note
            self.status.configure(text=text)
            self.apply_button.configure(state='normal')
            self.add_color_button.configure(state='normal' if len(self.palette_colors) < 256 else 'disabled')
            self.delete_color_button.configure(state='normal' if len(self.palette_colors) > 1 else 'disabled')
        self._jobs.submit(compute, done,
                          lambda error: self.status.configure(text=str(error)))

    def _start_color_pick(self, picker):
        self._color_picker = picker
        picker.grab_release()
        picker.withdraw()
        self.grab_set()
        for view in (self.source_view, self.result_view):
            view.pick_callback = self._finish_color_pick
            view.configure(cursor='crosshair')
        self.status.configure(text='Click either image to pick a color. Esc cancels picking.')

    def _finish_color_pick(self, color=None):
        picker, self._color_picker = self._color_picker, None
        for view in (self.source_view, self.result_view):
            view.pick_callback = None
            view.configure(cursor='')
        if picker is not None:
            if color is not None:
                picker.set_rgb(color)
            picker.deiconify()
            picker.lift()
            picker.grab_set()
        self.status.configure(text='')

    def _escape(self, _event=None):
        if self._color_picker is not None:
            self._finish_color_pick()
        else:
            self.close()

    def _poll_snapper(self):
        self._poll_id = None
        if not self._snap_closed:
            self._jobs.poll()
            self._poll_id = self.after(30, self._poll_snapper)

    def _apply(self):
        if self.preview_result is None:
            return
        try:
            if self._read_settings() != self._preview_settings:
                self._settings_changed()
                return
        except ValueError:
            self._settings_changed()
            return
        self.result = self.preview_result
        self.result_image = self.result.image
        self.result_plan = self._preview_video_plan
        if self.config_mgr:
            self.config_mgr.set('pixel_snapper', 'auto_colors', value=self.auto_colors_var.get())
            for key in ('num_colors', 'quantizer', 'pixel_size'):
                self.config_mgr.set('pixel_snapper', key, value=getattr(self._preview_settings, key))
            self.config_mgr.save()
        self.close()

    def close(self):
        self.grab_release()
        self.destroy()

    def destroy(self):
        self._snap_closed = True
        if hasattr(self, '_jobs'):
            self._jobs.close()
        if self._poll_id is not None:
            self.after_cancel(self._poll_id)
            self._poll_id = None
        for token in getattr(self, '_startup_ids', []):
            self.after_cancel(token)
        if hasattr(self, '_startup_ids'):
            self._startup_ids.clear()
        super().destroy()


class PixelSnapperDialog(SnapperContents, ctk.CTkToplevel):
    def __init__(self, parent, source, config=None, video_path=None):
        super().__init__(parent)
        self.title('Pixel Snapper')
        self.geometry('1100x650')
        self.minsize(900, 560)
        self.transient(parent)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.body = ctk.CTkFrame(self, fg_color='transparent')
        self.body.grid(row=0, column=0, sticky='nsew', padx=14, pady=14)
        self.footer = ctk.CTkFrame(self, fg_color='transparent')
        self.footer.grid(row=1, column=0, sticky='ew', padx=14, pady=(0, 14))
        self._initialize_snapper(source, config, video_path)
        self.grab_set()

