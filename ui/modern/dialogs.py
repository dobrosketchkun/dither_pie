"""Searchable actions and an algorithm gallery rendered by the real library."""
from functools import lru_cache
import tkinter as tk

import customtkinter as ctk
from PIL import Image

from dithering_lib import DitherMode, ImageDitherer
from ui.preview_jobs import LatestPreviewJob
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
    width, height = 48, 32
    source = Image.new('RGB', (width, height))
    source.putdata([(int(x * 255 / (width - 1)),) * 3 for y in range(height) for x in range(width)])
    images = []
    for mode in DitherMode:
        result = ImageDitherer(num_colors=2, dither_mode=mode,
            palette=[(0, 0, 0), (255, 255, 255)],
            dither_params={'size': 32} if mode == DitherMode.BLUE_NOISE else {}).apply_dithering(source)
        images.append((mode.value, result.resize((192, 128), Image.Resampling.NEAREST)))
    return images


class ModeGallery(Modal):
    def __init__(self, app):
        super().__init__(app, 'Dithering algorithms', '740x580')
        self.app = app
        self._jobs = LatestPreviewJob()
        self._poll_id = None
        self._images = []
        Label(self, 'Algorithms · preview samples', 'h1').pack(anchor='w', padx=20, pady=14)
        self.status = Label(self, 'Rendering examples…', 'small')
        self.status.pack(anchor='w', padx=20)
        self.body = ctk.CTkScrollableFrame(self, fg_color='transparent')
        self.body.pack(fill='both', expand=True, padx=14, pady=10)
        for col in range(3):
            self.body.grid_columnconfigure(col, weight=1)
        self._jobs.submit(gallery_images, self._ready, lambda e: self.status.configure(text=str(e)))
        self._poll()
        self.show()

    def _poll(self):
        self._jobs.poll()
        self._poll_id = self.after(40, self._poll)

    def _ready(self, images):
        self.status.configure(text='Default settings; blue-noise sample uses a smaller 32×32 matrix.')
        for i, (name, image) in enumerate(images):
            photo = ctk.CTkImage(light_image=image, dark_image=image, size=(192, 128))
            self._images.append(photo)
            Btn(self.body, name, image=photo, compound='top', height=164,
                command=lambda n=name: self._choose(n)).grid(row=i // 3, column=i % 3, padx=5, pady=5)

    def _choose(self, name):
        self.app.dither_mode.set(name)
        self.app._on_dither_mode_changed(name)
        self.close()

    def close(self):
        self._jobs.close()
        if self._poll_id is not None:
            self.after_cancel(self._poll_id)
            self._poll_id = None
        super().close()


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
