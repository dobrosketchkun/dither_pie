"""Prototype-style editor controls backed by the production editing canvas."""
import tkinter as tk
import customtkinter as ctk
from ui.pixelization_editor import PixelizationEditorDialog, PixelizationEditorCanvas
from . import theme as T, winchrome
from .modal import Dialog
from .widgets import Btn, Card, Label, SliderField


class ModernPixelizationEditor(winchrome.StableTitlebar, PixelizationEditorDialog):
    _apply_chrome = Dialog._apply_chrome
    _drag_press = Dialog._drag_press
    _drag_motion = Dialog._drag_motion
    _drag_release = Dialog._drag_release

    def _enable_maximize_box(self):
        self._apply_chrome()

    def _build_ui(self):
        self.configure(fg_color=T.c('bg'))
        self.minsize(850, 580)
        self._resizable = True
        self._drag_from = self._manual_drag = None
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self._head = ctk.CTkFrame(self, fg_color='transparent')
        self._head.grid(row=0, column=0, sticky='ew', padx=18, pady=(12, 8))
        self._head.grid_columnconfigure(1, weight=1)
        Label(self._head, 'Pixelization editor', 'h1').grid(row=0, column=0)
        self._close_btn = Btn(self._head, T.ICON['win_close'], width=32, command=self._on_cancel)
        self._close_btn.grid(row=0, column=2)
        body = ctk.CTkFrame(self, fg_color='transparent')
        body.grid(row=1, column=0, sticky='nsew', padx=14)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)
        sidebar = ctk.CTkScrollableFrame(body, width=260, fg_color='transparent')
        sidebar.grid(row=0, column=0, sticky='ns', padx=(0, 12))
        self.canvas = PixelizationEditorCanvas(body, bg=self._apply_appearance_mode(T.c('card_alt')), highlightthickness=0)
        self.canvas.grid(row=0, column=1, sticky='nsew')
        self.canvas.set_source_image(self.source_image)
        self.canvas.set_image(self.source_image, update=False)
        self.canvas.set_mode('preview')
        self.canvas.set_show_grid(True)

        grid = Card(sidebar, 'Grid')
        grid.pack(fill='x', pady=(0, 10))
        self.target_field = SliderField(grid.body, 'Shorter edge', 8, 256,
            self._get_config_value('target_size', 64), step=1, typed_bounds=(1, None),
            command=lambda _: self._apply_target_size())
        self.target_field.pack(fill='x')
        self.target_size_entry = self.target_field.entry
        self.grid_label = Label(grid.body, '', 'mono_s')
        self.grid_label.pack(anchor='w', pady=8)
        Label(grid.body, 'Position the source in the grid,\nthen pixelize to enable editing.', 'small').pack(anchor='w', pady=(0, 8))
        self.fit_to_grid_button = Btn(grid.body, 'Fit source to grid', command=self.canvas.fit_image_to_grid)
        self.fit_to_grid_button.pack(fill='x', pady=(0, 6))
        self.convert_button = Btn(grid.body, 'Pixelize', command=self._start_conversion)
        self.convert_button.pack(fill='x')

        tools = Card(sidebar, 'Tools')
        tools.pack(fill='x', pady=(0, 10))
        row = ctk.CTkFrame(tools.body, fg_color='transparent')
        row.pack(fill='x')
        self.tool_buttons = {}
        for col, (name, text) in enumerate((('brush', 'Brush'), ('magic', 'Wand'), ('picker', 'Pick'))):
            row.grid_columnconfigure(col, weight=1)
            button = Btn(row, text, width=1, command=lambda n=name: self._set_tool(n))
            button.grid(row=0, column=col, sticky='ew', padx=2)
            self.tool_buttons[name] = button
        self._tool_default_fg = {name: b.cget('fg_color') for name, b in self.tool_buttons.items()}
        self.tool_size_var = tk.StringVar(value=str(self._get_config_value('tool_size', 1)))
        self.size_field = SliderField(tools.body, 'Brush / wand size', 1, 16, int(self.tool_size_var.get()),
            step=1, typed_bounds=(1, None), command=self._size_changed)
        self.size_field.pack(fill='x', pady=10)
        self.tool_size_entry = self.size_field.entry
        self.color_button = Btn(tools.body, 'Drawing color', command=self._pick_color)
        self.color_button.pack(fill='x', pady=(0, 10))
        self._set_draw_color_from_hex('#000000')
        self.magic_threshold = tk.IntVar(value=self._get_config_value('magic_wand_threshold', 5))
        self.threshold_field = SliderField(tools.body, 'Wand threshold', 0, 30, self.magic_threshold.get(),
            step=1, command=self._apply_magic_threshold)
        self.threshold_field.pack(fill='x')
        self.magic_slider = self.threshold_field.slider
        self.magic_label = self.threshold_field.value_label
        self._magic_label_color = self.magic_label.cget('text_color')

        view = Card(sidebar, 'View & history')
        view.pack(fill='x')
        self.grid_toggle_var = tk.BooleanVar(value=self._get_config_value('grid_visible', True))
        self.grid_toggle = ctk.CTkSwitch(view.body, text='Pixel grid', variable=self.grid_toggle_var,
            command=self._apply_grid_toggle, progress_color=T.accent(), font=T.f('body'))
        self.grid_toggle.pack(anchor='w', pady=(0, 10))
        Btn(view.body, 'Fit view', command=self.canvas.fit_to_window).pack(fill='x', pady=(0, 8))
        history = ctk.CTkFrame(view.body, fg_color='transparent')
        history.pack(fill='x')
        self.undo_button = Btn(history, 'Undo', width=1, command=self._undo)
        self.undo_button.pack(side='left', fill='x', expand=True, padx=(0, 4))
        self.redo_button = Btn(history, 'Redo', width=1, command=self._redo)
        self.redo_button.pack(side='left', fill='x', expand=True)
        footer = ctk.CTkFrame(self, fg_color='transparent')
        footer.grid(row=2, column=0, sticky='ew', padx=18, pady=12)
        Label(footer, 'Scroll to zoom · drag to position · Alt to pick a color', 'small').pack(side='left')
        self.apply_button = Btn(footer, 'Apply', command=self._on_apply)
        self.apply_button.pack(side='right', padx=(8, 0))
        self.cancel_button = Btn(footer, 'Cancel', command=self._on_cancel)
        self.cancel_button.pack(side='right')
        self._set_edit_controls_state(False)
        self._set_tool(self._get_config_value('tool', 'brush'))
        self._bind_shortcuts()
        self.bind('<Escape>', lambda _: self._on_cancel())

    def _size_changed(self, value):
        self.tool_size_var.set(str(value))
        self._apply_tool_size()

    def _set_tool(self, tool):
        super()._set_tool(tool)
        for name, button in self.tool_buttons.items():
            button.configure(fg_color=T.accent_soft() if name == tool else T.c('card_alt'),
                             text_color=T.accent_text() if name == tool else T.c('text'))
        state = 'normal' if tool == 'magic' and self.canvas.mode == 'edit' else 'disabled'
        self.magic_slider.configure(state=state)
        self.threshold_field.entry.configure(state=state)

    def _set_edit_controls_state(self, enabled):
        super()._set_edit_controls_state(enabled)
        self.size_field.slider.configure(state='normal' if enabled else 'disabled')
        if not enabled:
            self.threshold_field.entry.configure(state='disabled')
