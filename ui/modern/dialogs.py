"""Searchable actions and an algorithm gallery rendered by the real library."""
from functools import lru_cache
from pathlib import Path
import tkinter as tk

import customtkinter as ctk
from PIL import Image

from dithering_lib import DitherMode
from .modal import Dialog
from . import theme as T, winchrome
from .widgets import Btn, Label, make_entry


class Modal(winchrome.StableTitlebar, ctk.CTkToplevel):
    def __init__(self, app, title, size):
        super().__init__(app)
        self.title(title)
        self.geometry(size)
        self.configure(fg_color=T.c('bg'))
        self.transient(app)
        self.protocol('WM_DELETE_WINDOW', self.close)
        self.bind('<Escape>', lambda _: self.close())

    def show(self):
        self.wait_visibility()
        self.grab_set()

    def close(self):
        self.grab_release()
        self.destroy()


@lru_cache(maxsize=1)
def gallery_images():
    images = []
    for mode in DitherMode:
        with Image.open(Path(__file__).parent / 'assets' / 'gallery' / (mode.value + '.png')) as sample:
            images.append((mode.value, sample.copy()))
    return images


class ModeGallery(Dialog):
    def __init__(self, app):
        super().__init__(app, 'Dithering algorithms', size=(740, 580), resizable=True)
        self.app = app
        self._images = []
        self.status = Label(self.footer, 'Default settings; blue-noise sample uses a 32×32 matrix.', 'small')
        self.status.pack(anchor='w')
        self.body.grid_rowconfigure(0, weight=1)
        self.samples = ctk.CTkScrollableFrame(self.body, fg_color='transparent')
        self.samples.grid(row=0, column=0, sticky='nsew')
        for col in range(3):
            self.samples.grid_columnconfigure(col, weight=1)
        self._ready(gallery_images())

    def _ready(self, images):
        for i, (name, image) in enumerate(images):
            photo = ctk.CTkImage(light_image=image, dark_image=image, size=(192, 128))
            self._images.append(photo)
            Btn(self.samples, name, image=photo, compound='top', height=164,
                command=lambda n=name: self._choose(n)).grid(row=i // 3, column=i % 3, padx=5, pady=5)

    def _choose(self, name):
        self.app.dither_mode.set(name)
        self.app._on_dither_mode_changed(name)
        self.close()

class CommandPalette(Modal):
    def __init__(self, app):
        super().__init__(app, 'Commands', '500x480')
        self.commands = [
            ('Open image', app.load_image), ('Open video', app.load_video),
            ('Random video frame', app.load_random_frame), ('Pixelize', app._on_pixelize_regular),
            ('Neural pixelize', app._on_pixelize_neural), ('Pixelization editor', app._on_pixelization_editor),
            ('Update preview', app._on_apply_dithering), ('Save image', app.save_image),
            ('Process video', app._apply_to_video_workflow), ('Fit image', app.fit_to_window),
            ('Before / after', app.toggle_compare), ('Preferences', app.open_settings)]
        self.entry = make_entry(self, placeholder='Search commands')
        self.entry.pack(fill='x', padx=16, pady=16)
        self.body = ctk.CTkScrollableFrame(self, fg_color='transparent')
        self.body.pack(fill='both', expand=True, padx=12, pady=(0, 12))
        self.entry.bind('<KeyRelease>', lambda _: self._filter())
        self.entry.bind('<Return>', self._first)
        self._filter()
        self.show()
        self.entry.focus_set()

    def _filter(self):
        query = self.entry.get().lower()
        self.filtered = [(name, action) for name, action in self.commands if query in name.lower()]
        for child in self.body.winfo_children():
            child.destroy()
        for name, action in self.filtered:
            Btn(self.body, name, command=lambda f=action: self._run(f)).pack(fill='x', pady=3)

    def _run(self, action):
        self.close()
        action()

    def _first(self, _event=None):
        if self.filtered:
            self._run(self.filtered[0][1])
        return 'break'
