"""The prototype's palette layout, backed by real palette data."""
import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk
from ui.palettes import CustomPaletteCreator
from . import theme as T
from .library import LIBRARY_PALETTES
from .processing import GENERATED, generate_palette
from .widgets import Btn, Icon, IconBtn, Label, PaletteCard, Segmented, Toggle


class PalettePanel:
    def _build_palette(self):
        panel = self.panels[2]
        body = self._card(2, 'Colors', 'sliders')
        self.colors_entry = self._number(body, 'Palette size', 2, 64,
            self.config.get('defaults', 'num_colors'), 1, (1, 256), self._palette_count_changed).entry
        self.gamma = Toggle(body, 'Gamma correction', 'Mix colors in linear light.',
            self.config.get('defaults', 'use_gamma'), command=self._request_preview)
        self.gamma.grid(row=2, column=0, sticky='ew', pady=(6, 0))
        self.gamma_var = self.gamma.var
        self.palette_var = tk.StringVar(value=GENERATED[0])
        self.palette_var.trace_add('write', self._palette_changed)
        tools = ctk.CTkFrame(panel, fg_color='transparent')
        tools.grid(row=2, column=0, sticky='ew', pady=(0, 8))
        tools.grid_columnconfigure(0, weight=1)
        search = ctk.CTkFrame(tools, fg_color=T.c('card'), corner_radius=T.R_CTRL,
                             border_width=1, border_color=T.c('border'))
        search.grid(row=0, column=0, sticky='ew')
        search.grid_columnconfigure(1, weight=1)
        Icon(search, 'search', 13, T.c('faint')).grid(row=0, column=0, padx=(10, 0))
        self.palette_search = ctk.CTkEntry(search, placeholder_text='Search palettes', width=1,
            border_width=0, fg_color='transparent', height=32, font=T.f('body'), text_color=T.c('text'))
        self.palette_search.grid(row=0, column=1, sticky='ew', padx=(6, 8))
        self.palette_search.bind('<KeyRelease>', self._schedule_palette_search)
        for col, (icon, callback, tip) in enumerate([
            ('plus', lambda: CustomPaletteCreator(self, self.palette_mgr, self._refresh_palettes), 'New custom palette'),
            ('link', lambda: self._import_from_lospec(self, self._refresh_palettes), 'Import from Lospec'),
            ('dropper', lambda: self._create_palette_from_image(self, self._refresh_palettes, self.palette_var), 'Extract from image')], 1):
            IconBtn(tools, icon, callback, size=32, tooltip=tip).grid(row=0, column=col, padx=(4, 0))
        self.palette_filter = Segmented(panel, ['All', 'Generated', 'Library', 'Mine'],
                                        command=lambda _: self._filter_palettes(), height=30)
        self.palette_filter.grid(row=3, column=0, sticky='ew', pady=(0, 10))
        self.palette_list = ctk.CTkFrame(panel, fg_color='transparent')
        self.palette_list.grid(row=4, column=0, sticky='ew')
        self.palette_list.grid_columnconfigure(0, weight=1)
        self.palette_cards = {}
        footer = ctk.CTkFrame(panel, fg_color='transparent')
        footer.grid(row=5, column=0, sticky='ew', pady=12)
        footer.grid_columnconfigure(0, weight=1)
        Btn(footer, 'Update preview', variant='primary', command=self._on_apply_dithering).grid(row=0, column=0, sticky='ew')
        self.delete_palette_button = Btn(footer, 'Delete selected custom palette', variant='subtle', command=self._delete_palette)
        self.delete_palette_button.grid(row=1, column=0, sticky='ew', pady=(6, 0))
        self._generated_palettes = {}
        self._palette_key = None
        self._refresh_palettes()

    def _palette_count_changed(self, *_):
        self._refresh_palettes()
        self._request_preview()

    def _palette_entries(self):
        saved = self.palette_mgr.list_palette_names()
        return ([(name, 'generated', self._generated_palettes.get(name)) for name in GENERATED]
                + [(name, 'library', colors) for name, _, colors in LIBRARY_PALETTES if name not in saved]
                + [(name, 'custom', self.palette_mgr.get_palette_colors_rgb(name)) for name in saved])

    def _selected_palette_colors(self):
        name = self.palette_var.get()
        if name in GENERATED:
            return None
        return next((colors for n, _, colors in self._palette_entries() if n == name), None)

    def _refresh_palettes(self):
        source = self.pixelized_image or self.current_image
        count = int(self.colors_entry.get())
        key = (id(source), count)
        if key != self._palette_key:
            self._palette_key = key
            self._generated_palettes = {}
            self._palette_jobs.invalidate()
            if source is not None:
                def compute():
                    return {name: generate_palette(source, name, count) for name in GENERATED}
                def done(palettes):
                    self._generated_palettes = palettes
                    self._filter_palettes()
                self._palette_jobs.submit(compute, done, lambda e: self.set_status(str(e), 'err'))
        if self.palette_var.get() not in [name for name, _, _ in self._palette_entries()]:
            self.palette_var.set(GENERATED[0])
        self._filter_palettes()

    def _schedule_palette_search(self, _event=None):
        if self._palette_search_after is not None:
            self.after_cancel(self._palette_search_after)
        self._palette_search_after = self.after(120, self._filter_palettes)

    def _filter_palettes(self):
        self._palette_search_after = None
        entries = self._palette_entries()
        query = self.palette_search.get().strip().lower()
        group = self.palette_filter.get()
        signature = (query, group, repr(entries))
        if signature == getattr(self, '_palette_list_signature', None):
            return
        self._palette_list_signature = signature
        for child in self.palette_list.winfo_children():
            child.destroy()
        self.palette_cards = {}
        groups = {'Generated': 'generated', 'Library': 'library', 'Mine': 'custom'}
        for name, badge, colors in entries:
            if query not in name.lower() or (group != 'All' and groups[group] != badge):
                continue
            card = PaletteCard(self.palette_list, name, colors or [], badge, self.palette_var.set,
                placeholder='Generating swatches…' if self.current_image is not None else 'Load a source image')
            card.grid(row=len(self.palette_cards), column=0, sticky='ew', pady=(0, 7))
            card.set_selected(name == self.palette_var.get())
            self.palette_cards[name] = card
        if not self.palette_cards:
            Label(self.palette_list, 'No palette matches that.', 'small', T.c('faint')).grid(row=0, column=0, pady=16)
        self._update_delete_palette()

    def _update_delete_palette(self):
        self.delete_palette_button.configure(state='normal' if self.palette_mgr.get_palette(self.palette_var.get()) else 'disabled')

    def _palette_changed(self, *_):
        for name, card in getattr(self, 'palette_cards', {}).items():
            card.set_selected(name == self.palette_var.get())
        self._update_delete_palette()
        self._request_preview(activate=True)

    def _delete_palette(self):
        name = self.palette_var.get()
        if self.palette_mgr.get_palette(name) and messagebox.askyesno('Delete palette', f'Delete {name}?', parent=self):
            self.palette_mgr.remove_palette(name)
            self._refresh_palettes()
